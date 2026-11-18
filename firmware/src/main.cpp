#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"
#include "img_converters.h"

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

// ---------- WiFi settings ----------
#define USE_AP false // true = ESP32 makes its own WiFi network
const char *WIFI_SSID = "Group4";
const char *WIFI_PASS = "piGroup4";
const char *AP_SSID = "ESP32-Digits";
const char *AP_PASS = "12345678"; // at least 8 characters

#define FLASH_LED 4

WebServer server(80);

void fail(const char *msg)
{
    Serial.println(msg);
    while (true)
        delay(1000);
}

// ---------- Camera (AI-Thinker pins) ----------
bool initCamera()
{
    camera_config_t c = {};
    c.ledc_channel = LEDC_CHANNEL_0;
    c.ledc_timer = LEDC_TIMER_0;
    c.pin_d0 = 5;
    c.pin_d1 = 18;
    c.pin_d2 = 19;
    c.pin_d3 = 21;
    c.pin_d4 = 36;
    c.pin_d5 = 39;
    c.pin_d6 = 34;
    c.pin_d7 = 35;
    c.pin_xclk = 0;
    c.pin_pclk = 22;
    c.pin_vsync = 25;
    c.pin_href = 23;
    c.pin_sccb_sda = 26;
    c.pin_sccb_scl = 27;
    c.pin_pwdn = 32;
    c.pin_reset = -1;
    c.xclk_freq_hz = 20000000;
    c.pixel_format = PIXFORMAT_GRAYSCALE;
    c.frame_size = FRAMESIZE_QQVGA; // 160 x 120
    c.fb_count = 2;
    c.fb_location = CAMERA_FB_IN_PSRAM;
    c.grab_mode = CAMERA_GRAB_LATEST;
    if (esp_camera_init(&c) != ESP_OK)
        return false;

    sensor_t *s = esp_camera_sensor_get();
    s->set_brightness(s, 2); // max sensor brightness
    return true;
}

// ---------- TFLite Micro ----------
constexpr int kArenaSize = 16 * 1024;
alignas(16) static uint8_t tensor_arena[kArenaSize];

tflite::MicroErrorReporter error_reporter;
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

// ---------- Image buffers ----------
constexpr int FW = 160, FH = 120; // camera frame
constexpr int IMG = 20;           // model input 20x20
constexpr int SCALE = 7;          // 20x20 -> 140x140 for viewing
constexpr int BIG = IMG * SCALE;
constexpr int BORDER = 8; // ignore dark frame edges when finding the digit

uint8_t *frame = nullptr;    // raw camera frame        (PSRAM)
uint8_t *ink = nullptr;      // white digit on black    (PSRAM)
uint8_t *inputBig = nullptr; // enlarged 20x20 for page (PSRAM)
float img[IMG * IMG];        // model input 0..1

String predText = "Press c to predict";

// ---------- Step 1: get a frame ----------
bool grabFrame()
{
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb)
        return false;
    bool ok = (fb->width == FW && fb->height == FH);
    if (ok)
        memcpy(frame, fb->buf, FW * FH);
    esp_camera_fb_return(fb);
    return ok;
}

// ---------- Step 2: frame -> white digit on black (also the live view) ----------
bool makeInk()
{
    int mn = 255, mx = 0;
    for (int i = 0; i < FW * FH; i++)
    {
        int v = 255 - frame[i]; // invert: dark ink -> bright
        mn = min(mn, v);
        mx = max(mx, v);
    }
    if (mx - mn < 40)
    { // flat image, nothing written
        memset(ink, 0, FW * FH);
        return false;
    }
    int th = mn + (mx - mn) * 4 / 10; // above this = ink
    for (int i = 0; i < FW * FH; i++)
    {
        int v = 255 - frame[i];
        ink[i] = (v > th) ? (uint8_t)((v - th) * 255 / (mx - th)) : 0;
    }
    return true;
}

// ---------- Step 3: find digit, center it, shrink to 20x20 ----------
bool preprocess()
{
    int x0 = FW, y0 = FH, x1 = -1, y1 = -1;
    for (int y = BORDER; y < FH - BORDER; y++)
        for (int x = BORDER; x < FW - BORDER; x++)
            if (ink[y * FW + x] > 0)
            {
                x0 = min(x0, x);
                x1 = max(x1, x);
                y0 = min(y0, y);
                y1 = max(y1, y);
            }
    if (x1 < 0)
        return false;
    int bw = x1 - x0 + 1, bh = y1 - y0 + 1;
    if (max(bw, bh) < 10)
        return false; // too small = noise

    int side = max(bw, bh) * 13 / 10; // square box + margin
    if (side < IMG)
        side = IMG;
    int cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    int sx = cx - side / 2, sy = cy - side / 2;

    float peak = 0;
    for (int r = 0; r < IMG; r++)
    {
        for (int c = 0; c < IMG; c++)
        {
            int ys = sy + r * side / IMG, ye = sy + (r + 1) * side / IMG;
            int xs = sx + c * side / IMG, xe = sx + (c + 1) * side / IMG;
            float sum = 0;
            int count = 0;
            for (int y = ys; y < ye; y++)
                for (int x = xs; x < xe; x++)
                {
                    count++;
                    if (x < 0 || y < 0 || x >= FW || y >= FH)
                        continue; // outside = black
                    sum += ink[y * FW + x] / 255.0f;
                }
            float v = count ? sum / count : 0;
            img[r * IMG + c] = v;
            peak = max(peak, v);
        }
    }
    if (peak > 0)
        for (int i = 0; i < IMG * IMG; i++)
            img[i] /= peak; // brightest = 1.0
    return true;
}

void printAscii()
{
    for (int r = 0; r < IMG; r++)
    {
        for (int c = 0; c < IMG; c++)
        {
            float v = img[r * IMG + c];
            Serial.print(v > 0.6f ? '#' : (v > 0.2f ? '+' : '.'));
        }
        Serial.println();
    }
}

// ---------- Step 4: capture -> predict (only when "c" is pressed) ----------
void doPredict()
{
    if (!grabFrame())
    {
        predText = "Capture failed";
        return;
    }
    makeInk();
    if (!preprocess())
    {
        memset(img, 0, sizeof(img));
        predText = "No digit found";
        Serial.println(predText);
        return;
    }

    uint32_t t0 = millis();
    for (int i = 0; i < IMG * IMG; i++)
        input->data.f[i] = img[i];
    if (interpreter->Invoke() != kTfLiteOk)
    {
        predText = "Invoke failed";
        return;
    }
    uint32_t dt = millis() - t0;

    // scores -> probabilities (softmax only if the model doesn't do it)
    float p[10], sum = 0;
    for (int i = 0; i < 10; i++)
    {
        p[i] = output->data.f[i];
        sum += p[i];
    }
    if (fabs(sum - 1.0f) > 0.01f)
    {
        float m = p[0], s = 0;
        for (int i = 1; i < 10; i++)
            m = max(m, p[i]);
        for (int i = 0; i < 10; i++)
        {
            p[i] = expf(p[i] - m);
            s += p[i];
        }
        for (int i = 0; i < 10; i++)
            p[i] /= s;
    }

    // top 3 guesses
    int idx[10];
    for (int i = 0; i < 10; i++)
        idx[i] = i;
    for (int i = 0; i < 3; i++)
        for (int j = i + 1; j < 10; j++)
            if (p[idx[j]] > p[idx[i]])
            {
                int t = idx[i];
                idx[i] = idx[j];
                idx[j] = t;
            }

    predText = "Digit " + String(idx[0]) + " (" + String((int)(p[idx[0]] * 100)) + "%)" + "  |  next: " + String(idx[1]) + " (" + String((int)(p[idx[1]] * 100)) + "%), " + String(idx[2]) + " (" + String((int)(p[idx[2]] * 100)) + "%)" + "  |  " + String(dt) + " ms";

    Serial.println(predText);
    printAscii();
    Serial.println();
}

// ---------- Web server ----------
void sendGrayJpeg(const uint8_t *buf, int w, int h)
{
    uint8_t *jpg = nullptr;
    size_t len = 0;
    if (!fmt2jpg((uint8_t *)buf, w * h, w, h, PIXFORMAT_GRAYSCALE, 80, &jpg, &len))
    {
        server.send(500, "text/plain", "JPEG failed");
        return;
    }
    server.sendHeader("Cache-Control", "no-store");
    server.setContentLength(len);
    server.send(200, "image/jpeg", "");
    server.client().write(jpg, len);
    free(jpg);
}

void handleRoot()
{
    server.send(200, "text/html", R"(
<html><body style="background:#111;color:#eee;font-family:sans-serif;text-align:center">
<h3>Live view (white digit on black = what the model gets)</h3>
<img id="cam" width="480"><br>
<button id="btn" style="font-size:22px;padding:10px 30px;margin:12px">Predict (c)</button>
<h2 id="res">Press c to predict</h2>
<h3>Model input (20x20)</h3>
<img id="inp" width="200" style="image-rendering:pixelated">
<script>
// live feed: ask for the next frame only after the last one arrived
function nextFrame() { cam.src = '/cam.jpg?' + Date.now(); }
cam.onload  = () => setTimeout(nextFrame, 100);
cam.onerror = () => setTimeout(nextFrame, 500);
nextFrame();

let last = '';
function show(t) {
  res.innerText = t;
  if (t !== last) { last = t; inp.src = '/input.jpg?' + Date.now(); }
}
function predict() {
  res.innerText = 'Predicting...';
  fetch('/predict').then(r => r.text()).then(show);
}
btn.onclick = predict;
document.onkeydown = e => { if (e.key === 'c' || e.key === 'C') predict(); };

// also pick up predictions triggered from the Serial Monitor
setInterval(() => fetch('/pred').then(r => r.text()).then(show), 1000);
</script>
</body></html>)");
}

void handleCam()
{ // live processed view
    if (grabFrame())
        makeInk();
    sendGrayJpeg(ink, FW, FH);
}

void handleInput()
{ // 20x20 model input, enlarged
    for (int y = 0; y < BIG; y++)
        for (int x = 0; x < BIG; x++)
            inputBig[y * BIG + x] = (uint8_t)(img[(y / SCALE) * IMG + (x / SCALE)] * 255);
    sendGrayJpeg(inputBig, BIG, BIG);
}

void handlePredict()
{ // "c" from the browser
    doPredict();
    server.send(200, "text/plain", predText);
}

void handlePred()
{ // latest result
    server.send(200, "text/plain", predText);
}

void startWifi()
{
    if (USE_AP)
    {
        WiFi.softAP(AP_SSID, AP_PASS);
        Serial.printf("WiFi network: %s  ->  open http://%s\n",
                      AP_SSID, WiFi.softAPIP().toString().c_str());
    }
    else
    {
        WiFi.begin(WIFI_SSID, WIFI_PASS);
        Serial.print("Connecting to WiFi");
        while (WiFi.status() != WL_CONNECTED)
        {
            delay(500);
            Serial.print(".");
        }
        Serial.printf("\nOpen http://%s\n", WiFi.localIP().toString().c_str());
    }
    server.on("/", handleRoot);
    server.on("/cam.jpg", handleCam);
    server.on("/input.jpg", handleInput);
    server.on("/predict", handlePredict);
    server.on("/pred", handlePred);
    server.begin();
}

// ---------- Setup ----------
void setup()
{
    Serial.begin(115200);
    delay(500);

    frame = (uint8_t *)ps_malloc(FW * FH);
    ink = (uint8_t *)ps_malloc(FW * FH);
    inputBig = (uint8_t *)ps_malloc(BIG * BIG);
    if (!frame || !ink || !inputBig)
        fail("PSRAM alloc failed");
    memset(ink, 0, FW * FH);

    if (!initCamera())
        fail("Camera init failed");
    Serial.println("Camera OK");

    pinMode(FLASH_LED, OUTPUT);
    digitalWrite(FLASH_LED, HIGH); // LED on after the camera has started

    const tflite::Model *model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION)
        fail("Schema mismatch");
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kArenaSize, &error_reporter);
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk)
        fail("AllocateTensors failed");
    input = interpreter->input(0);
    output = interpreter->output(0);
    if (!input || !output)
        fail("input/output is null");
    Serial.println("Model OK");

    startWifi();
    Serial.println("Ready. Press 'c' here or in the browser to predict.");
}

// ---------- Loop ----------
void loop()
{
    server.handleClient(); // live view + browser commands

    if (Serial.available())
    { // "c" from the Serial Monitor
        char ch = Serial.read();
        if (ch == 'c' || ch == 'C')
            doPredict();
    }
}
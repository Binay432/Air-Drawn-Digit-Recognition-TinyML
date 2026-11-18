#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ---------------- Model selection ----------------
#include "model_data_8.h"
#define MODEL_ARRAY g_model_8
#define MODEL_LEN g_model_8_len
#define MODEL_NAME "INT8"

// ---------------- Config ----------------
#define SCREEN_W 128
#define SCREEN_H 64
#define OLED_ADDR 0x3C
#define ARENA_KB 60 // tune after seeing "Arena used" on Serial

Adafruit_SSD1306 display(SCREEN_W, SCREEN_H, &Wire, -1);

// ---------------- OLED helpers ----------------
void showDigit(int digit)
{
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.print("Predicted digit:");
    display.setTextSize(5); // 30x40 px per char
    display.setCursor(49, 18);
    display.print(digit);
    display.display();
}

void oled(const String &l1, const String &l2 = "", const String &l3 = "")
{
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.print(l1);
    display.setCursor(0, 20);
    display.print(l2);
    display.setCursor(0, 40);
    display.print(l3);
    display.display();
}

// ---------------- TFLite Micro ----------------
constexpr int kArenaSize = ARENA_KB * 1024;
alignas(16) static uint8_t tensor_arena[kArenaSize];

tflite::MicroErrorReporter error_reporter;
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

void fail(const char *msg)
{
    Serial.println(msg);
    oled("ERROR", msg);
    while (true)
        delay(1000);
}

void printTensorInfo(const char *name, TfLiteTensor *t)
{
    Serial.printf("%s: type=%s dims=[", name, TfLiteTypeGetName(t->type));
    for (int i = 0; i < t->dims->size; i++)
        Serial.printf("%d%s", t->dims->data[i], i < t->dims->size - 1 ? "," : "");
    Serial.printf("] bytes=%u scale=%.8f zero_point=%d\n",
                  (unsigned)t->bytes, t->params.scale, (int)t->params.zero_point);
}

// ---------------- Setup ----------------
void setup()
{
    Serial.begin(115200);
    delay(500);
    Wire.begin(21, 22); // SDA, SCL

    if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR))
    {
        Serial.println("OLED not found - check wiring/address");
        while (true)
            delay(1000);
    }
    display.setTextColor(SSD1306_WHITE);
    oled("Loading model...", MODEL_NAME);

    // 1. Load model from flash
    Serial.printf("Model: %s, size: %u bytes\n", MODEL_NAME, (unsigned)MODEL_LEN);
    const tflite::Model *model = tflite::GetModel(MODEL_ARRAY);
    if (model->version() != TFLITE_SCHEMA_VERSION)
        fail("Schema mismatch");

    // 2. Build interpreter + allocate tensors in the arena
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kArenaSize, &error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
        fail("AllocateTensors failed");

    // 3. Grab input/output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);

    printTensorInfo("Input ", input);
    printTensorInfo("Output", output);
    Serial.printf("Arena used: %u / %d bytes\n",
                  (unsigned)interpreter->arena_used_bytes(), kArenaSize);

    // // 4. Sanity check for the 8-bit model
    // if (input->type != kTfLiteInt8)
    //     Serial.println("WARNING: input is not int8 - check the conversion");

    oled("Model ready", MODEL_NAME,
         "Arena " + String(interpreter->arena_used_bytes() / 1024) + "/" + String(ARENA_KB) + "KB");
}

void loop()
{
    delay(1000); // inference on test samples comes in step 3
}
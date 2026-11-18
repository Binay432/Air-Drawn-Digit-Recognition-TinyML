#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data_8.h"
#include "test_data.h"

#define MODEL_ARRAY g_model_8
#define MODEL_LEN g_model_8_len
#define MODEL_NAME "INT8"

// ---------------- Oled Config ----------------
#define SCREEN_W 128
#define SCREEN_H 64
#define OLED_ADDR 0x3C
#define ARENA_KB 60 // tunable

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
constexpr int kArenaSize = ARENA_KB * 1024; // working memory for the model
alignas(16) static uint8_t tensor_arena[kArenaSize];

tflite::MicroErrorReporter error_reporter;       // tflite print own error message to serial
tflite::AllOpsResolver resolver;                 // resolver holds the actual code for cnn operation (conv2d, maxpool, dense and softmax)
tflite::MicroInterpreter *interpreter = nullptr; // engine that runs the model
TfLiteTensor *input = nullptr;                   // pointer to models input data
TfLiteTensor *output = nullptr;                  // pointer to models output data

void fail(const char *msg)
{
    Serial.println(msg);
    oled("ERROR", msg);
    while (true)
        delay(1000);
}

// To see the  input, outpur shape
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

    // Check data has loaded
    // Test data check
    Serial.printf("Test samples: %d x %d\n", kNumSamples, kInputSize);
    Serial.printf("Test data size: %u bytes\n", (unsigned)sizeof(test_samples));
    Serial.printf("Labels size: %u bytes\n", (unsigned)sizeof(test_labels));

    oled("Model ready", MODEL_NAME,
         "Arena " + String(interpreter->arena_used_bytes() / 1024) + "/" + String(ARENA_KB) + "KB");
}

void loop()
{
    static bool done = false; // for single run
    if (done)
    {
        delay(1000);
        return;
    }

    int correct = 0;
    unsigned long totalTime = 0;

    Serial.printf("\nTesting %d samples...\n", kNumSamples);
    Serial.println("Wrong predictions:");

    for (int i = 0; i < kNumSamples; i++)
    {
        // 1. Copy one sample into the model input
        for (int k = 0; k < kInputSize; k++)
            input->data.f[k] = test_samples[i][k];

        // 2. Inference run and prediction time
        unsigned long t0 = micros();
        if (interpreter->Invoke() != kTfLiteOk)
            fail("Invoke failed");
        totalTime += micros() - t0;

        // 3. Argmax: class with the highest score
        int pred = 0;
        float best = output->data.f[0];
        for (int c = 1; c < 10; c++)
        {
            if (output->data.f[c] > best)
            {
                best = output->data.f[c];
                pred = c;
            }
        }

        // 4. Count correct, print only the wrong ones
        if (pred == test_labels[i])
            correct++;

        // 5. Progress on OLED every 20 samples
        if (i % 20 == 0)
            oled("Testing 8-bit...", String(i) + " / " + String(kNumSamples));
    }

    // 6. Final result
    float acc = (correct / kNumSamples) * 100.0f;
    unsigned long avgTime = totalTime / kNumSamples;

    Serial.printf("\nAccuracy: %d/%d = %.2f%%\n", correct, kNumSamples, acc);
    Serial.printf("Avg inference time: %lu us\n", avgTime);
    Serial.printf("Total inference time: %lu ms\n", totalTime / 1000);

    oled("8-bit model",
         "Acc: " + String(correct) + "/" + String(kNumSamples),
         String(acc, 2) + "%  " + String(avgTime) + " us");

    done = true;
}
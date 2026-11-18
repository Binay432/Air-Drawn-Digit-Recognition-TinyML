#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

#define SCREEN_W 128
#define SCREEN_H 64
#define OLED_ADDR 0x3C
#define KARENA_SIZE 60 // tune as go

Adafruit_SSD1306 display(SCREEN_W, SCREEN_H, &Wire, -1);

void showDigit(int digit)
{
    display.clearDisplay();

    display.setTextSize(1);
    display.setCursor(0, 0);
    display.print("Predicted digit:");

    display.setTextSize(5);    // each character is 30x40 pixels
    display.setCursor(49, 18); // roughly centered
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

// TFlite micro
constexpr int kArenaSize = KARENA_SIZE * 1024; // working memory for the model
alignas(16) static uint8_t tensor_arena[kArenaSize];

tflite::MicroErrorReporter error_reporter;       // tflite print own error message to serial
tflite::AllOpsResolver resolver;                 // resolver holds the actual code for cnn operation (conv2d, maxpool, dense and softmax)
tflite::MicroInterpreter *interpreter = nullptr; // engine that runs the model
TfLiteTensor *input = nullptr;                   // pointer to models input data
TfLiteTensor *output = nullptr;                  // pointer to models output data

// helper
void fail(const char *msg)
{
    Serial.println(msg);
    oled("ERROR", msg);
    while (true)
        delay(1000);
}

void setup()
{
    // ---------- Start hardware ----------
    Serial.begin(115200);
    delay(500);
    Wire.begin(21, 22); // SDA, SCL
    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C))
    {
        Serial.println("OLED not found"); // keep going, Serial still works
    }
    display.setTextColor(SSD1306_WHITE);
    oled("Loading model...");

    // ---------- Check 1: model readable ----------
    Serial.printf("Model size: %u bytes\n", g_model_len);
    const tflite::Model *model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION)
        fail("Schema version mismatch");
    Serial.println("Check 1 OK: model readable");

    // ---------- Check 2: memory setup ----------
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kArenaSize, &error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
        fail("AllocateTensors failed");
    Serial.println("Check 2 OK: AllocateTensors");
    Serial.printf("Arena used: %u of %d bytes\n",
                  interpreter->arena_used_bytes(), kArenaSize);

    // ---------- Check 3: input/output details ----------
    input = interpreter->input(0);
    output = interpreter->output(0);
    if (input == nullptr || output == nullptr)
        fail("input/output is null");

    Serial.print("Input shape: [");
    for (int i = 0; i < input->dims->size; i++)
    {
        Serial.print(input->dims->data[i]);
        if (i < input->dims->size - 1)
            Serial.print(", ");
    }
    Serial.println("]");

    Serial.print("Output shape: [");
    for (int i = 0; i < output->dims->size; i++)
    {
        Serial.print(output->dims->data[i]);
        if (i < output->dims->size - 1)
            Serial.print(", ");
    }
    Serial.println("]");

    Serial.printf("Input type: %s\n", input->type == kTfLiteInt8 ? "int8" : "NOT int8");
    Serial.printf("Input scale: %f, zero point: %d\n",
                  input->params.scale, input->params.zero_point);
    Serial.println("Check 3 OK: input/output found");

    // ---------- Check 4: one test inference on a blank image ----------
    for (int i = 0; i < input->bytes; i++)
    {
        input->data.int8[i] = input->params.zero_point; // zero_point = "pixel value 0"
    }

    uint32_t t0 = micros();
    if (interpreter->Invoke() != kTfLiteOk)
        fail("Invoke failed");
    uint32_t dt = micros() - t0;

    int num_classes = output->dims->data[output->dims->size - 1];
    int best = 0;
    for (int i = 1; i < num_classes; i++)
    {
        if (output->data.int8[i] > output->data.int8[best])
            best = i;
    }
    Serial.printf("Check 4 OK: inference took %u us, predicted %d (blank image)\n", dt, best);

    // ---------- Summary on OLED ----------
    oled("Model deployed OK",
         "Arena: " + String(interpreter->arena_used_bytes() / 1024) + " KB",
         "Infer: " + String(dt / 1000.0, 1) + " ms");
}
void loop()
{
    // Demo: count 0-9 so you can see the display works
    for (int d = 0; d <= 9; d++)
    {
        showDigit(d);
        delay(1000);
    }
}
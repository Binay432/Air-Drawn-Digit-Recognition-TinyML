#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_W 128
#define SCREEN_H 64
#define OLED_ADDR 0x3C

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

void setup()
{
    Serial.begin(115200);
    Wire.begin(21, 22); // SDA, SCL

    if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR))
    {
        Serial.println("OLED not found - check wiring/address");
        while (true)
            delay(1000);
    }
    display.setTextColor(SSD1306_WHITE);
    display.clearDisplay();
    display.setTextSize(2);
    display.setCursor(0, 20);
    display.print("Hello Bin!");
    display.display();
    delay(2000);
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
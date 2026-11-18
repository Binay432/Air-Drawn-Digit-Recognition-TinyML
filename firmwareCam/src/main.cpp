#include <Arduino.h>
#include "esp_camera.h"

#define FLASH_LED 4
#define LINK_RX 14 // from WROOM TX2 (GPIO 17)
#define LINK_TX 13 // to   WROOM RX2 (GPIO 16)

HardwareSerial Link(1);

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
  s->set_brightness(s, 2); // max brightness
  return true;
}

void setup()
{
  Serial.begin(115200);

  if (!initCamera())
  {
    Serial.println("Camera init failed");
    while (true)
      delay(1000);
  }

  pinMode(FLASH_LED, OUTPUT);
  digitalWrite(FLASH_LED, HIGH); // LED on only after the camera works

  Link.begin(921600, SERIAL_8N1, LINK_RX, LINK_TX);
  Serial.println("Camera ready, waiting for requests");
}
void loop()
{
  if (!Link.available())
    return;
  if (Link.read() != 'C')
    return; // only react to a capture request

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb)
    return;

  const uint8_t header[4] = {0xAA, 0x55, 0xAA, 0x55};
  uint16_t w = fb->width, h = fb->height;
  Link.write(header, 4);        // start marker
  Link.write((uint8_t *)&w, 2); // width
  Link.write((uint8_t *)&h, 2); // height
  Link.write(fb->buf, fb->len); // pixels (160*120 bytes)

  esp_camera_fb_return(fb);
}

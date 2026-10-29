// This sketch is designed for an ESP8266 (NodeMCU) with a 2x8 LED matrix (16x8 pixels).
// paste it into the Arduino IDE and upload it to your ESP8266 board.
// install the MD_MAX72xx library via the Arduino Library Manager before uploading.
// install the SPI library via the Arduino Library Manager before uploading.
// select the write port and board type in the Arduino IDE before uploading.
// -Deep 
#include <MD_MAX72xx.h>
#include <SPI.h>

#define HARDWARE_TYPE MD_MAX72XX::FC16_HW
#define MAX_DEVICES 2

#define DATA_PIN D7
#define CLK_PIN  D5
#define CS_PIN D8

MD_MAX72XX matrix(
  HARDWARE_TYPE,
  DATA_PIN,
  CLK_PIN,
  CS_PIN,
  MAX_DEVICES
);

String input = "";

// Set one LED using transposed logical coordinates.
// Original grid: x = 0..15, y = 0..7.
// Transposed grid: x = 0..7, y = 0..15.
void setTransposedPixel(uint8_t x, uint8_t y, bool on) {
  if (x >= 16 || y >= 8) return;

  uint8_t tx = y;
  uint8_t ty = x;

  // Transposed coordinates are 8 columns x 16 rows.
  // Map each group of 8 rows to one physical matrix.
  uint8_t device = ty / 8;
  uint8_t row = ty % 8;
  uint8_t col = tx;

  matrix.setPoint(row, device * 8 + col, on);
}

// Apply a 16-column mask.
// '1' means OFF, '0' means ON.
void applyMask(String mask) {
  if (mask.length() != 16) return;

  matrix.clear();

  for (uint8_t x = 0; x < 16; x++) {
    bool off = mask.charAt(x) == '1';

    // Each logical column is represented by all 8 pixels.
    for (uint8_t y = 0; y < 8; y++) {
      setTransposedPixel(x, y, !off);
    }
  }
}

void setup() {
  Serial.begin(115200);

  matrix.begin();
  matrix.control(MD_MAX72XX::INTENSITY, 2);
  matrix.clear();

  Serial.println("Transposed calibration ready");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (input.startsWith("M:")) {
        applyMask(input.substring(2));
      }
      input = "";
    } else {
      input += c;

      if (input.length() > 32) input = "";
    }
  }
}

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <DFRobot_BMI160.h>
#include <DFRobot_GNSS.h>

const char* ssid = "SSID";
const char* password = "password";

String scriptURL = "urlapps.script";

#define SDA_PIN 21
#define SCL_PIN 22
#define BMI160_ADDR 0x69
#define GNSS_ADDR  0x20

DFRobot_BMI160 bmi160;
DFRobot_GNSS_I2C gnss(&Wire, GNSS_ADDR);

int16_t accel[3];
int16_t gyro[3];

float lat = 0, lon = 0;

unsigned long prevSensorMillis = 0;
unsigned long prevSendMillis = 0;
const unsigned long SENSOR_INTERVAL = 500;
const unsigned long SEND_INTERVAL = 1000;

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  delay(300);

  if (bmi160.softReset() != 0) Serial.println("BMI160 reset failed!");
  if (bmi160.I2cInit(BMI160_ADDR) != 0) {
    Serial.println("BMI160 I2C init failed!");
    while (1);
  }
  Serial.println("BMI160 OK.");

  if (!gnss.begin()) {
    Serial.println("GNSS init failed!");
    while (1);
  }
  Serial.println("GNSS OK.");

  Serial.println("Connecting WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected.");
}

void loop() {
  unsigned long now = millis();

  if (now - prevSensorMillis >= SENSOR_INTERVAL) {
    prevSensorMillis = now;

    bmi160.getAccelData(accel);
    bmi160.getGyroData(gyro);

    sLonLat_t latitude = gnss.getLat();
    sLonLat_t longitude = gnss.getLon();

    lat = latitude.latDD;
    lon = longitude.lonDDD;

    Serial.printf("ACCEL: %d %d %d | GYRO: %d %d %d\n",
                  accel[0], accel[1], accel[2],
                  gyro[0], gyro[1], gyro[2]);

    Serial.printf("GPS: lat=%.6f lon=%.6f\n", lat, lon);
  }

  if (now - prevSendMillis >= SEND_INTERVAL) {
    prevSendMillis = now;

    if (WiFi.status() == WL_CONNECTED) {
      HTTPClient http;

      http.begin(scriptURL);
      http.addHeader("Content-Type", "application/json");

      String mapsLink = "https://www.google.com/maps?q=" + 
                        String(lat, 6) + "," + 
                        String(lon, 6);

      String json = "{";
      json += "\"timestamp\":\"" + String(now) + "\",";
      json += "\"accel_x\":\"" + String(accel[0]) + "\",";
      json += "\"accel_y\":\"" + String(accel[1]) + "\",";
      json += "\"accel_z\":\"" + String(accel[2]) + "\",";
      json += "\"gyro_x\":\"" + String(gyro[0]) + "\",";
      json += "\"gyro_y\":\"" + String(gyro[1]) + "\",";
      json += "\"gyro_z\":\"" + String(gyro[2]) + "\",";
      json += "\"lat\":\"" + String(lat, 6) + "\",";
      json += "\"lon\":\"" + String(lon, 6) + "\",";
      json += "\"maps\":\"" + mapsLink + "\"";
      json += "}";

      int httpCode = http.POST(json);
      Serial.printf("Upload Result: %d\n", httpCode);

      if (httpCode > 0) Serial.println("Data terkirim ke Google Sheet.");
      else Serial.println("Gagal mengirim.");

      http.end();
    }
  }
}

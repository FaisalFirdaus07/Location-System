#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <DFRobot_BNO055.h> 
#include <DFRobot_GNSS.h>   
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

// --- KONFIGURASI JARINGAN ---
const char* ssid     = "ssid";      
const char* password = "pass";  
String scriptURL     = "appscripturl"; 

// --- PIN I2C (Sesuai dengan example yang berhasil) ---
#define I2C_SDA 5
#define I2C_SCL 4

// --- DEKLARASI OBJEK SENSOR ---
typedef DFRobot_BNO055_IIC BNO;
BNO bno(&Wire, 0x28); 
DFRobot_GNSS_I2C gnss(&Wire, 0x20);

// --- VARIABEL GLOBAL (Shared Data) ---
SemaphoreHandle_t dataMutex;

BNO::sAxisAnalog_t shared_accel;
BNO::sAxisAnalog_t shared_gyro;
BNO::sAxisAnalog_t shared_mag;
BNO::sEulAnalog_t  shared_euler;

double shared_lat, shared_lon, shared_alt;
uint8_t shared_satellites;
unsigned long shared_timestamp;

const unsigned long UPLOAD_INTERVAL = 1000; 

// --- FUNGSI PELACAK ERROR BNO055 ---
void printLastOperateStatus(BNO::eStatus_t eStatus) {
  switch(eStatus) {
    case BNO::eStatusOK:   Serial.println(" segalanya OK"); break;
    case BNO::eStatusErr:  Serial.println(" error tak diketahui"); break;
    case BNO::eStatusErrDeviceNotDetect:   Serial.println(" perangkat tak terdeteksi"); break;
    case BNO::eStatusErrDeviceReadyTimeOut:    Serial.println(" device ready timeout (masih loading)"); break;
    case BNO::eStatusErrDeviceStatus:    Serial.println(" device internal status error"); break;
    default: Serial.println(" status tak diketahui"); break;
  }
}

// --- CORE 0: TASK UPLOAD ---
void TaskUpload(void *pvParameters) {
  unsigned long lastUpload = 0;

  while (true) {
    if (millis() - lastUpload >= UPLOAD_INTERVAL) {
      lastUpload = millis();

      if (WiFi.status() != WL_CONNECTED) {
        vTaskDelay(100 / portTICK_PERIOD_MS); 
        continue;
      }

      BNO::sAxisAnalog_t tx_accel, tx_gyro, tx_mag;
      BNO::sEulAnalog_t  tx_euler;
      double tx_lat, tx_lon, tx_alt;
      uint8_t tx_sat;
      unsigned long tx_time;

      if (xSemaphoreTake(dataMutex, (TickType_t)10) == pdTRUE) {
        tx_accel = shared_accel;
        tx_gyro  = shared_gyro;
        tx_mag   = shared_mag;
        tx_euler = shared_euler;
        tx_lat   = shared_lat;
        tx_lon   = shared_lon;
        tx_alt   = shared_alt;
        tx_sat   = shared_satellites;
        tx_time  = shared_timestamp;
        xSemaphoreGive(dataMutex); 
      } else {
        continue;
      }

      HTTPClient http;
      http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
      http.begin(scriptURL);
      http.addHeader("Content-Type", "application/json");

      String json = "{";
      json += "\"timestamp\":" + String(tx_time) + ",";
      json += "\"ax\":" + String(tx_accel.x) + ",\"ay\":" + String(tx_accel.y) + ",\"az\":" + String(tx_accel.z) + ",";
      json += "\"gx\":" + String(tx_gyro.x) + ",\"gy\":" + String(tx_gyro.y) + ",\"gz\":" + String(tx_gyro.z) + ",";
      json += "\"mx\":" + String(tx_mag.x) + ",\"my\":" + String(tx_mag.y) + ",\"mz\":" + String(tx_mag.z) + ",";
      json += "\"head\":" + String(tx_euler.head) + ",\"pitch\":" + String(tx_euler.pitch) + ",\"roll\":" + String(tx_euler.roll) + ",";
      json += "\"lat\":" + String(tx_lat, 8) + ",\"lon\":" + String(tx_lon, 8) + ",";
      json += "\"alt\":" + String(tx_alt, 2) + ",\"sat\":" + String(tx_sat);
      json += "}";

      int httpCode = http.POST(json);
      if (httpCode > 0) {
        Serial.println("[Upload] ✅ Data Terkirim ke Spreadsheet!");
      }
      http.end();
    }
    vTaskDelay(10 / portTICK_PERIOD_MS); 
  }
}

// --- SETUP ---
void setup() {
  Serial.begin(115200);

  // Menggunakan pin I2C persis seperti example
  Wire.begin(I2C_SDA, I2C_SCL);
  dataMutex = xSemaphoreCreateMutex();

  Serial.println("\n--- Memulai Inisialisasi BNO055 ---");

  bno.reset();
  // SISTEM RETRY SEPERTI KODE EXAMPLE
  while(bno.begin() != BNO::eStatusOK) {
    Serial.print("⚠️ Menunggu BNO055 siap...");
    printLastOperateStatus(bno.lastOperateStatus);
    delay(2000); // Tunggu 2 detik lalu coba lagi
  }
  Serial.println("✅ BNO055 Berhasil Terhubung!");

  Serial.println("\n--- Memulai Inisialisasi GNSS ---");
  if (!gnss.begin()) {
    Serial.println("⚠️ GNSS Gagal Terhubung (Pastikan kabel GPS tidak terbalik)");
  } else {
    gnss.enablePower();
    Serial.println("✅ GNSS Berhasil Terhubung!");
  }

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.println("⏳ Menghubungkan WiFi di background...");

  xTaskCreatePinnedToCore(TaskUpload, "UploadTask", 10000, NULL, 1, NULL, 0);
}

// --- CORE 1: MAIN LOOP ---
void loop() {
  BNO::sAxisAnalog_t raw_accel = bno.getAxis(BNO::eAxisAcc);
  BNO::sAxisAnalog_t raw_gyro  = bno.getAxis(BNO::eAxisGyr);
  BNO::sAxisAnalog_t raw_mag   = bno.getAxis(BNO::eAxisMag);
  BNO::sEulAnalog_t  raw_euler = bno.getEul(); 

  sLonLat_t latVal = gnss.getLat();
  sLonLat_t lonVal = gnss.getLon();
  double raw_lat = latVal.latitudeDegree;
  double raw_lon = lonVal.lonitudeDegree;

  if(raw_lat > 0 && raw_lat < 10) raw_lat = -raw_lat;

  if (xSemaphoreTake(dataMutex, (TickType_t)5) == pdTRUE) {
    shared_accel = raw_accel;
    shared_gyro  = raw_gyro;
    shared_mag   = raw_mag;
    shared_euler = raw_euler;
    shared_lat   = raw_lat;
    shared_lon   = raw_lon;
    shared_alt   = gnss.getAlt();
    shared_satellites = gnss.getNumSatUsed();
    shared_timestamp  = millis();
    xSemaphoreGive(dataMutex);
  }

  static unsigned long prevPrint = 0;
  if (millis() - prevPrint > 500) { 
    prevPrint = millis();
    Serial.printf("[Sensor] Yaw:%.1f° | Pitch:%.1f° | Roll:%.1f° | Lat:%.6f | Sat:%d\n", 
                  raw_euler.head, raw_euler.pitch, raw_euler.roll, raw_lat, shared_satellites);
  }
}

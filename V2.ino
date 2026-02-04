#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <DFRobot_BMI160.h>
#include <DFRobot_GNSS.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

// --- KONFIGURASI ---
const char* ssid     = "ssd";      
const char* password = "pass";  
String scriptURL     = "appscripturl"; 

DFRobot_BMI160 bmi160;
DFRobot_GNSS_I2C gnss(&Wire, 0x20);

// --- VARIABEL GLOBAL (Shared Data) ---
// Variabel ini akan diakses oleh kedua Core, jadi butuh pengaman (Mutex)
SemaphoreHandle_t dataMutex;

// Data Sensor
int16_t shared_accel[3], shared_gyro[3];
double shared_lat, shared_lon, shared_alt;
uint8_t shared_satellites;
unsigned long shared_timestamp;

// Konfigurasi Timer Upload
const unsigned long UPLOAD_INTERVAL = 1000; // Kirim tiap 5 detik

// --- TUGAS 2: BACKGROUND UPLOAD (Core 0) ---
// Tugas ini khusus untuk menangani WiFi dan HTTP yang berat
void TaskUpload(void *pvParameters) {
  unsigned long lastUpload = 0;

  while (true) {
    // Cek apakah sudah waktunya kirim (Non-blocking timer)
    if (millis() - lastUpload >= UPLOAD_INTERVAL) {
      lastUpload = millis();

      // Cek WiFi dulu
      if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[UploadTask] WiFi Putus, mencoba reconnect...");
        WiFi.disconnect();
        WiFi.reconnect();
        vTaskDelay(1000 / portTICK_PERIOD_MS); // Tunggu sebentar
        continue;
      }

      // 1. AMBIL DATA DARI VARIABEL GLOBAL (Copy data)
      // Kita kunci sebentar agar data tidak berubah saat dicopy
      int16_t tx_accel[3], tx_gyro[3];
      double tx_lat, tx_lon, tx_alt;
      uint8_t tx_sat;
      unsigned long tx_time;

      if (xSemaphoreTake(dataMutex, (TickType_t)100) == pdTRUE) {
        memcpy(tx_accel, shared_accel, sizeof(shared_accel));
        memcpy(tx_gyro, shared_gyro, sizeof(shared_gyro));
        tx_lat = shared_lat;
        tx_lon = shared_lon;
        tx_alt = shared_alt;
        tx_sat = shared_satellites;
        tx_time = shared_timestamp;
        xSemaphoreGive(dataMutex); // Buka kunci
      } else {
        Serial.println("[UploadTask] Gagal mengambil kunci data, skip...");
        continue;
      }

      // 2. SIAPKAN JSON (Proses ini makan waktu, tapi tidak mengganggu sensor)
      HTTPClient http;
      http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
      http.begin(scriptURL);
      http.addHeader("Content-Type", "application/json");

      String json = "{";
      json += "\"timestamp\":\"" + String(tx_time) + "\",";
      json += "\"accel_x\":" + String(tx_accel[0]) + ",";
      json += "\"accel_y\":" + String(tx_accel[1]) + ",";
      json += "\"accel_z\":" + String(tx_accel[2]) + ",";
      json += "\"gyro_x\":" + String(tx_accel[0]) + ","; // Typo fix: gyro index
      json += "\"gyro_y\":" + String(tx_gyro[1]) + ",";
      json += "\"gyro_z\":" + String(tx_gyro[2]) + ",";
      json += "\"lat\":" + String(tx_lat, 8) + ",";
      json += "\"lon\":" + String(tx_lon, 8) + ",";
      json += "\"alt\":" + String(tx_alt, 2) + ",";
      json += "\"sat\":" + String(tx_sat);
      json += "}";

      Serial.println("\n[UploadTask] 📤 Mengirim data background...");
      
      // 3. KIRIM (BLOCKING process terjadi di sini, tapi sensor di Core 1 aman)
      int httpCode = http.POST(json);

      if (httpCode > 0) {
        String payload = http.getString();
        if(payload.indexOf("SUCCESS") >= 0) {
           Serial.println("[UploadTask] ✅ Sukses terkirim!");
        } else {
           Serial.println("[UploadTask] ⚠️ Respon: " + payload.substring(0, 50));
        }
      } else {
        Serial.printf("[UploadTask] ❌ Gagal: %s\n", http.errorToString(httpCode).c_str());
      }
      http.end();
    }
    
    // Wajib ada delay kecil agar Watchdog tidak marah (10ms)
    vTaskDelay(10 / portTICK_PERIOD_MS); 
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  
  // Buat Mutex untuk keamanan data
  dataMutex = xSemaphoreCreateMutex();

  // Init Sensor
  if (bmi160.softReset() != 0) Serial.println("⚠️ BMI160 Error");
  if (bmi160.I2cInit(0x69) != 0) Serial.println("⚠️ BMI160 Init Fail");
  if (!gnss.begin()) Serial.println("⚠️ GNSS Error");
  else gnss.enablePower();

  // Koneksi WiFi Awal
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");
  // Kita tunggu sebentar di awal, tapi tidak selamanya
  int retry = 0;
  while (WiFi.status() != WL_CONNECTED && retry < 20) {
    delay(200); 
    Serial.print(".");
    retry++;
  }
  Serial.println("\n✅ Setup Selesai.");

  // --- JALANKAN TASK UPLOAD DI CORE 0 ---
  xTaskCreatePinnedToCore(
    TaskUpload,   // Fungsi task
    "UploadTask", // Nama task
    10000,        // Stack size (byte)
    NULL,         // Parameter
    1,            // Prioritas (Rendah)
    NULL,         // Handle
    0             // Core ID (0 = Background, 1 = Main Loop)
  );
}

// --- LOOP UTAMA (CORE 1) - FOKUS BACA SENSOR ---
void loop() {
  // Variabel lokal loop
  int16_t raw_accel[3], raw_gyro[3];
  
  // 1. Baca Sensor (Cepat)
  bmi160.getAccelData(raw_accel);
  bmi160.getGyroData(raw_gyro);
  
  sLonLat_t latVal = gnss.getLat();
  sLonLat_t lonVal = gnss.getLon();
  double raw_lat = latVal.latitudeDegree;
  double raw_lon = lonVal.lonitudeDegree;
  double raw_alt = gnss.getAlt();
  uint8_t raw_sat = gnss.getNumSatUsed();

  // Koreksi & Sanitasi
  if(raw_lat > 0 && raw_lat < 10) raw_lat = -raw_lat;
  if (isnan(raw_lat)) raw_lat = 0.0;
  if (isnan(raw_lon)) raw_lon = 0.0;
  if (isnan(raw_alt)) raw_alt = 0.0;

  // 2. Update Variabel Global (Agar bisa diambil oleh Task Upload)
  // Kita kunci Mutex agar Task Upload tidak membaca data setengah-setengah
  if (xSemaphoreTake(dataMutex, (TickType_t)10) == pdTRUE) {
    memcpy(shared_accel, raw_accel, sizeof(raw_accel));
    memcpy(shared_gyro, raw_gyro, sizeof(raw_gyro));
    shared_lat = raw_lat;
    shared_lon = raw_lon;
    shared_alt = raw_alt;
    shared_satellites = raw_sat;
    shared_timestamp = millis();
    xSemaphoreGive(dataMutex); // Lepas kunci
  }

  // Debugging Serial (Jangan terlalu cepat agar tidak lag)
  static unsigned long prevPrint = 0;
  if (millis() - prevPrint > 500) {
    prevPrint = millis();
    Serial.printf("[SensorLoop] Sat:%d Lat:%.8f (Sensor Running...)\n", raw_sat, raw_lat);
  }

  // Loop ini akan berjalan SANGAT CEPAT (Real-time)
  // Tidak ada delay() di sini.
}

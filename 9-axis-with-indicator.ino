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
String scriptURL     = "urlappscript"; 

// --- PIN I2C ---
#define I2C_SDA 5
#define I2C_SCL 4

// --- PIN LED INDIKATOR ---
#define LED_GPS_STAT   10
#define LED_IMU_STAT   12
#define LED_DATA_UP    19
#define LED_BATT_LOW   16
#define LED_BATT_MED   17
#define LED_BATT_HIGH  18

// --- PIN SENSOR BATERAI (Opsional untuk Hardware) ---
// Ganti dengan pin analog ADC yang terhubung ke pembagi tegangan baterai fisik
#define PIN_BATT_ADC   36 

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

// --- FUNGSI INDIKATOR BATERAI ---
void updateBatteryLEDs() {
  // SIMULASI PEMBACAAN: Ubah analogRead jika sudah ada sirkuit baterai fisik
  // Asumsi baterai LiPo 3.7V (Max 4.2V, Min 3.2V)
  // Untuk saat ini kita pakai variabel dummy agar bisa di-compile.
  int battPercentage = 80; // Ganti logika ini dengan pembacaan ADC (analogRead)

  if (battPercentage >= 70) {
    digitalWrite(LED_BATT_HIGH, HIGH);
    digitalWrite(LED_BATT_MED, LOW);
    digitalWrite(LED_BATT_LOW, LOW);
  } 
  else if (battPercentage >= 30) {
    digitalWrite(LED_BATT_HIGH, LOW);
    digitalWrite(LED_BATT_MED, HIGH);
    digitalWrite(LED_BATT_LOW, LOW);
  } 
  else {
    digitalWrite(LED_BATT_HIGH, LOW);
    digitalWrite(LED_BATT_MED, LOW);
    digitalWrite(LED_BATT_LOW, HIGH);
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
      json += "\"gx\":" + String(tx_gyro.x)  + ",\"gy\":" + String(tx_gyro.y)  + ",\"gz\":" + String(tx_gyro.z)  + ",";
      json += "\"mx\":" + String(tx_mag.x)   + ",\"my\":" + String(tx_mag.y)   + ",\"mz\":" + String(tx_mag.z)   + ",";
      json += "\"head\":" + String(tx_euler.head) + ",\"pitch\":" + String(tx_euler.pitch) + ",\"roll\":" + String(tx_euler.roll) + ",";
      json += "\"lat\":" + String(tx_lat, 8) + ",\"lon\":" + String(tx_lon, 8) + ",";
      json += "\"alt\":" + String(tx_alt, 2) + ",\"sat\":" + String(tx_sat);
      json += "}";

      int httpCode = http.POST(json);
      if (httpCode > 0) {
        Serial.println("[Upload] ✅ Data Terkirim ke Spreadsheet!");
        
        // --- KEDIPKAN LED DATA UP (Sukses Kirim) ---
        digitalWrite(LED_DATA_UP, HIGH);
        vTaskDelay(100 / portTICK_PERIOD_MS); // Nyala 100ms
        digitalWrite(LED_DATA_UP, LOW);
      } else {
        digitalWrite(LED_DATA_UP, LOW); // Pastikan mati jika gagal
      }
      http.end();
    }
    vTaskDelay(10 / portTICK_PERIOD_MS); 
  }
}

// --- SETUP ---
void setup() {
  Serial.begin(115200);

  // --- INIT PIN LED ---
  pinMode(LED_GPS_STAT, OUTPUT);
  pinMode(LED_IMU_STAT, OUTPUT);
  pinMode(LED_DATA_UP, OUTPUT);
  pinMode(LED_BATT_LOW, OUTPUT);
  pinMode(LED_BATT_MED, OUTPUT);
  pinMode(LED_BATT_HIGH, OUTPUT);

  // Matikan semua LED di awal
  digitalWrite(LED_GPS_STAT, LOW);
  digitalWrite(LED_IMU_STAT, LOW);
  digitalWrite(LED_DATA_UP, LOW);
  
  // Cek Status Baterai (Simulasi awal)
  updateBatteryLEDs();

  Wire.begin(I2C_SDA, I2C_SCL);
  dataMutex = xSemaphoreCreateMutex();

  Serial.println("\n--- Memulai Inisialisasi BNO055 ---");

  bno.reset();
  
  // LED IMU Berkedip selama proses inisialisasi
  while(bno.begin() != BNO::eStatusOK) {
    Serial.print("⚠️ Menunggu BNO055 siap...");
    printLastOperateStatus(bno.lastOperateStatus);
    
    digitalWrite(LED_IMU_STAT, !digitalRead(LED_IMU_STAT)); // Blink
    delay(2000);
  }
  Serial.println("✅ BNO055 Berhasil Terhubung!");
  digitalWrite(LED_IMU_STAT, HIGH); // LED Solid (Standby/OK)

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

  uint8_t current_satellites = gnss.getNumSatUsed();

  // --- UPDATE LED GPS ---
  if (current_satellites > 0) {
    digitalWrite(LED_GPS_STAT, HIGH); // Sinyal Fix
  } else {
    digitalWrite(LED_GPS_STAT, LOW);  // Mencari Satelit
  }

  if (xSemaphoreTake(dataMutex, (TickType_t)5) == pdTRUE) {
    shared_accel = raw_accel;
    shared_gyro  = raw_gyro;
    shared_mag   = raw_mag;
    shared_euler = raw_euler;
    shared_lat   = raw_lat;
    shared_lon   = raw_lon;
    shared_alt   = gnss.getAlt();
    shared_satellites = current_satellites;
    shared_timestamp  = millis();
    xSemaphoreGive(dataMutex);
  }

  static unsigned long prevPrint = 0;
  if (millis() - prevPrint > 500) { 
    prevPrint = millis();
    Serial.printf("[Sensor] Yaw:%.1f° Pitch:%.1f° Roll:%.1f°\n",
                  raw_euler.head, raw_euler.pitch, raw_euler.roll);
    Serial.printf("         Accel(m/s²) X:%.2f Y:%.2f Z:%.2f\n",
                  raw_accel.x, raw_accel.y, raw_accel.z);
    Serial.printf("         Gyro(°/s)   X:%.2f Y:%.2f Z:%.2f\n",
                  raw_gyro.x, raw_gyro.y, raw_gyro.z);
    Serial.printf("         Lat:%.6f Lon:%.6f Sat:%d\n",
                  raw_lat, raw_lon, current_satellites);
                  
    // Panggil update LED baterai secara periodik
    updateBatteryLEDs(); 
  }
}

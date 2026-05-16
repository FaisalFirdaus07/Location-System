#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <DFRobot_BNO055.h> 
#include <DFRobot_GNSS.h>   
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <time.h>           // Untuk NTP & Unix timestamp
#include <sys/time.h>

// --- KONFIGURASI JARINGAN ---
const char* ssid     = "ssid";      
const char* password = "pass";  
String scriptURL     = "appscripturl"; 

// --- KONFIGURASI NTP ---
#define NTP_SERVER      "pool.ntp.org"
#define GMT_OFFSET_SEC  25200   // WIB = UTC+7 (7 * 3600)
#define DST_OFFSET_SEC  0

// --- PIN I2C ---
#define I2C_SDA 5
#define I2C_SCL 4

// --- KONFIGURASI WIFI ---
#define WIFI_TIMEOUT_MS     10000
#define WIFI_RETRY_DELAY_MS 5000
#define WIFI_MAX_RETRY      5

// --- KONFIGURASI HTTP ---
#define HTTP_CONNECT_TIMEOUT_MS 800   // Batas waktu koneksi TCP
#define HTTP_RESPONSE_TIMEOUT_MS 900  // Batas waktu tunggu respons server

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
unsigned long long shared_timestamp_ms; // Unix timestamp dalam ms (seperti xlsx)
char shared_datetime[32];               // String datetime: "2026-05-16 14:30:00.123"

bool ntpSynced = false;

const unsigned long UPLOAD_INTERVAL = 1000;

// --- QUEUE: Buffer data antar Core (agar interval 1 detik tidak dipengaruhi durasi HTTP) ---
typedef struct {
  BNO::sAxisAnalog_t accel, gyro, mag;
  BNO::sEulAnalog_t  euler;
  double lat, lon, alt;
  uint8_t sat;
  unsigned long long timestamp_ms;
  char datetime[32];
} UploadPacket;

QueueHandle_t uploadQueue;

// --- FUNGSI PELACAK ERROR BNO055 ---
void printLastOperateStatus(BNO::eStatus_t eStatus) {
  switch(eStatus) {
    case BNO::eStatusOK:                   Serial.println(" segalanya OK"); break;
    case BNO::eStatusErr:                  Serial.println(" error tak diketahui"); break;
    case BNO::eStatusErrDeviceNotDetect:   Serial.println(" perangkat tak terdeteksi"); break;
    case BNO::eStatusErrDeviceReadyTimeOut:Serial.println(" device ready timeout (masih loading)"); break;
    case BNO::eStatusErrDeviceStatus:      Serial.println(" device internal status error"); break;
    default:                               Serial.println(" status tak diketahui"); break;
  }
}

// --- FUNGSI: Ambil Unix Timestamp dalam Milidetik (seperti kolom Timestamp xlsx) ---
unsigned long long getUnixTimestampMs() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (unsigned long long)(tv.tv_sec) * 1000ULL + (tv.tv_usec / 1000ULL);
}

// --- FUNGSI: Format Datetime String (seperti kolom Datetime xlsx) ---
void getDatetimeString(char* buf, size_t bufSize) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  struct tm timeinfo;
  localtime_r(&tv.tv_sec, &timeinfo);
  int ms = tv.tv_usec / 1000;
  snprintf(buf, bufSize, "%04d-%02d-%02d %02d:%02d:%02d.%03d",
    timeinfo.tm_year + 1900,
    timeinfo.tm_mon + 1,
    timeinfo.tm_mday,
    timeinfo.tm_hour,
    timeinfo.tm_min,
    timeinfo.tm_sec,
    ms);
}

// --- FUNGSI: Sync NTP ---
bool syncNTP() {
  Serial.println("[NTP] Menyinkronisasi waktu...");
  configTime(GMT_OFFSET_SEC, DST_OFFSET_SEC, NTP_SERVER);

  struct tm timeinfo;
  unsigned long start = millis();
  while (!getLocalTime(&timeinfo)) {
    if (millis() - start > 5000) {
      Serial.println("[NTP] ❌ Gagal sinkronisasi waktu!");
      return false;
    }
    vTaskDelay(500 / portTICK_PERIOD_MS);
  }

  char timeBuf[32];
  strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", &timeinfo);
  Serial.printf("[NTP] ✅ Waktu tersinkron: %s (WIB)\n", timeBuf);
  return true;
}

// --- FUNGSI KONEKSI WIFI (BLOCKING + TIMEOUT) ---
bool connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return true;

  Serial.printf("[WiFi] Menghubungkan ke '%s'...\n", ssid);
  WiFi.disconnect(true);
  delay(100);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  unsigned long startTime = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - startTime >= WIFI_TIMEOUT_MS) {
      Serial.println("[WiFi] ❌ Timeout! Gagal terhubung.");
      return false;
    }
    vTaskDelay(500 / portTICK_PERIOD_MS);
    Serial.print(".");
  }

  Serial.printf("\n[WiFi] ✅ Terhubung! IP: %s\n", WiFi.localIP().toString().c_str());

  // Sync NTP setelah WiFi berhasil konek (hanya sekali)
  if (!ntpSynced) {
    ntpSynced = syncNTP();
  }

  return true;
}

// --- FUNGSI RECONNECT WIFI DENGAN RETRY ---
bool ensureWiFiConnected() {
  if (WiFi.status() == WL_CONNECTED) return true;

  Serial.println("[WiFi] ⚠️ Koneksi terputus, mencoba reconnect...");

  for (int attempt = 1; attempt <= WIFI_MAX_RETRY; attempt++) {
    Serial.printf("[WiFi] Percobaan %d/%d...\n", attempt, WIFI_MAX_RETRY);
    if (connectWiFi()) return true;
    vTaskDelay(WIFI_RETRY_DELAY_MS / portTICK_PERIOD_MS);
  }

  Serial.println("[WiFi] ❌ Semua percobaan gagal. Mereset WiFi...");
  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);
  vTaskDelay(1000 / portTICK_PERIOD_MS);
  return connectWiFi();
}

// --- CORE 0: TASK UPLOAD ---
// Menerima data dari queue dan mengirimnya via HTTP.
// Durasi HTTP tidak mempengaruhi interval pengiriman karena
// jadwal pengumpulan data ada di loop() (Core 1), bukan di sini.
void TaskUpload(void *pvParameters) {
  UploadPacket pkt;

  while (true) {
    // Tunggu data baru dari queue (blocking, max 2 detik)
    if (xQueueReceive(uploadQueue, &pkt, pdMS_TO_TICKS(2000)) != pdTRUE) {
      continue; // Tidak ada data, tunggu lagi
    }

    if (!ensureWiFiConnected()) {
      Serial.println("[Upload] ⏭️ Skip upload, WiFi tidak tersedia.");
      continue;
    }

    HTTPClient http;
    http.setFollowRedirects(HTTPC_FORCE_FOLLOW_REDIRECTS); // Lebih cepat dari STRICT
    http.setConnectTimeout(HTTP_CONNECT_TIMEOUT_MS);        // Timeout koneksi TCP
    http.setTimeout(HTTP_RESPONSE_TIMEOUT_MS);              // Timeout tunggu respons
    http.begin(scriptURL);
    http.addHeader("Content-Type", "application/json");

    String json = "{";
    json += "\"datetime\":\"" + String(pkt.datetime) + "\",";
    json += "\"timestamp\":" + String((uint64_t)pkt.timestamp_ms) + ",";
    json += "\"ax\":" + String(pkt.accel.x) + ",\"ay\":" + String(pkt.accel.y) + ",\"az\":" + String(pkt.accel.z) + ",";
    json += "\"gx\":" + String(pkt.gyro.x) + ",\"gy\":" + String(pkt.gyro.y) + ",\"gz\":" + String(pkt.gyro.z) + ",";
    json += "\"mx\":" + String(pkt.mag.x) + ",\"my\":" + String(pkt.mag.y) + ",\"mz\":" + String(pkt.mag.z) + ",";
    json += "\"head\":" + String(pkt.euler.head) + ",\"pitch\":" + String(pkt.euler.pitch) + ",\"roll\":" + String(pkt.euler.roll) + ",";
    json += "\"lat\":" + String(pkt.lat, 8) + ",\"lon\":" + String(pkt.lon, 8) + ",";
    json += "\"alt\":" + String(pkt.alt, 2) + ",\"sat\":" + String(pkt.sat);
    json += "}";

    int httpCode = http.POST(json);
    if (httpCode > 0) {
      Serial.printf("[Upload] ✅ Terkirim! [%s] HTTP:%d\n", pkt.datetime, httpCode);
    } else {
      Serial.printf("[Upload] ❌ HTTP Error: %s\n", http.errorToString(httpCode).c_str());
    }
    http.end();
  }
}

// --- SETUP ---
void setup() {
  Serial.begin(115200);

  Wire.begin(I2C_SDA, I2C_SCL);
  dataMutex = xSemaphoreCreateMutex();

  // Inisialisasi queue (buffer 5 paket jika HTTP sempat lambat)
  uploadQueue = xQueueCreate(5, sizeof(UploadPacket));

  // Inisialisasi shared_datetime agar tidak kosong sebelum NTP sync
  strncpy(shared_datetime, "1970-01-01 00:00:00.000", sizeof(shared_datetime));
  shared_timestamp_ms = 0;

  Serial.println("\n--- Memulai Inisialisasi BNO055 ---");
  bno.reset();
  while(bno.begin() != BNO::eStatusOK) {
    Serial.print("⚠️ Menunggu BNO055 siap...");
    printLastOperateStatus(bno.lastOperateStatus);
    delay(2000);
  }
  Serial.println("✅ BNO055 Berhasil Terhubung!");

  Serial.println("\n--- Memulai Inisialisasi GNSS ---");
  if (!gnss.begin()) {
    Serial.println("⚠️ GNSS Gagal Terhubung (Pastikan kabel GPS tidak terbalik)");
  } else {
    gnss.enablePower();
    Serial.println("✅ GNSS Berhasil Terhubung!");
  }

  Serial.println("\n--- Memulai Koneksi WiFi + NTP ---");
  connectWiFi(); // Koneksi awal + sync NTP otomatis di dalamnya

  xTaskCreatePinnedToCore(TaskUpload, "UploadTask", 10000, NULL, 1, NULL, 0);
}

// --- CORE 1: MAIN LOOP ---
// Mengumpulkan data sensor setiap 1 detik tepat, lalu memasukkannya ke queue.
// Interval ini TIDAK dipengaruhi oleh durasi HTTP di TaskUpload (Core 0).
void loop() {
  static unsigned long lastCollect = 0;

  // Kumpulkan & antrekan data tepat setiap 1 detik
  if (millis() - lastCollect >= UPLOAD_INTERVAL) {
    lastCollect = millis();

    BNO::sAxisAnalog_t raw_accel = bno.getAxis(BNO::eAxisAcc);
    BNO::sAxisAnalog_t raw_gyro  = bno.getAxis(BNO::eAxisGyr);
    BNO::sAxisAnalog_t raw_mag   = bno.getAxis(BNO::eAxisMag);
    BNO::sEulAnalog_t  raw_euler = bno.getEul();

    sLonLat_t latVal = gnss.getLat();
    sLonLat_t lonVal = gnss.getLon();
    double raw_lat = latVal.latitudeDegree;
    double raw_lon = lonVal.lonitudeDegree;
    if (raw_lat > 0 && raw_lat < 10) raw_lat = -raw_lat;

    // Buat paket data untuk dikirim ke queue
    UploadPacket pkt;
    pkt.accel = raw_accel;
    pkt.gyro  = raw_gyro;
    pkt.mag   = raw_mag;
    pkt.euler = raw_euler;
    pkt.lat   = raw_lat;
    pkt.lon   = raw_lon;
    pkt.alt   = gnss.getAlt();
    pkt.sat   = gnss.getNumSatUsed();
    pkt.timestamp_ms = getUnixTimestampMs();
    getDatetimeString(pkt.datetime, sizeof(pkt.datetime));

    // Update shared data (untuk Serial print)
    if (xSemaphoreTake(dataMutex, (TickType_t)5) == pdTRUE) {
      shared_accel        = raw_accel;
      shared_gyro         = raw_gyro;
      shared_mag          = raw_mag;
      shared_euler        = raw_euler;
      shared_lat          = raw_lat;
      shared_lon          = raw_lon;
      shared_alt          = pkt.alt;
      shared_satellites   = pkt.sat;
      shared_timestamp_ms = pkt.timestamp_ms;
      strncpy(shared_datetime, pkt.datetime, sizeof(shared_datetime));
      xSemaphoreGive(dataMutex);
    }

    // Masukkan ke queue (non-blocking: jika queue penuh, data lama dibuang)
    if (xQueueSend(uploadQueue, &pkt, 0) != pdTRUE) {
      Serial.println("[Loop] ⚠️ Queue penuh, data dibuang (HTTP terlalu lambat)");
    }

    Serial.printf("[Sensor] %s | Yaw:%.1f° | Lat:%.6f | Sat:%d\n",
                  pkt.datetime, raw_euler.head, raw_lat, pkt.sat);
  }

  vTaskDelay(10 / portTICK_PERIOD_MS);
}

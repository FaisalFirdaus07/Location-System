// ============================================================
//  Inkubator 3T — ESP32 + BNO055 + GNSS + Firebase RTDB
//  v4: Heading Mirror Fix + IMU Zero-Offset saat GPS Fix
// ============================================================

#include <WiFi.h>
#include <Wire.h>
#include <DFRobot_BNO055.h>
#include <DFRobot_GNSS.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

// --- LIBRARY FIREBASE (Mobizt) ---
#include <Firebase_ESP_Client.h>
#include <addons/TokenHelper.h>
#include <addons/RTDBHelper.h>

// ============================================================
//  KONFIGURASI — SESUAIKAN BAGIAN INI
// ============================================================
const char* ssid     = "Damien";
const char* password = "faisal12";

#define API_KEY      "API_Key"
#define DATABASE_URL "URL"

// ============================================================
//  KOREKSI HEADING MIRROR
//
//  Jika BNO055 dipasang terbalik (misal flat menghadap bawah)
//  atau sumbu X/Y magnetometer terbalik, heading akan mirror:
//  Timur (90°) terbaca sebagai Barat (270°), dst.
//
//  Set true  → aktifkan koreksi (360 - heading)
//  Set false → nonaktifkan, pakai nilai asli BNO055
// ============================================================
#define HEADING_MIRROR_FIX  true

// ============================================================
//  KONFIGURASI NTP
// ============================================================
#define NTP_SERVER      "pool.ntp.org"
#define GMT_OFFSET_SEC   25200
#define DST_OFFSET_SEC   0
#define NTP_TIMEOUT_MS   8000

// ============================================================
//  PIN I2C
// ============================================================
#define I2C_SDA 5
#define I2C_SCL 4

// ============================================================
//  INTERVAL & BUFFER
// ============================================================
#define SENSOR_INTERVAL_MS      1000
#define QUEUE_SIZE                60
#define WIFI_CONNECT_TIMEOUT   15000
#define WIFI_RECONNECT_TIMEOUT 10000

// ============================================================
//  KONFIGURASI IMU ZERO-OFFSET
//  Offset di-latch saat GPS fix pertama kali diperoleh
// ============================================================
#define GPS_STABLE_COUNT   3    // Butuh N sampel GPS valid berturut-turut
#define GPS_LOSS_RESET_SEC 30   // Reset offset jika GPS hilang > N detik

// ============================================================
//  OBJEK SENSOR & FIREBASE
// ============================================================
typedef DFRobot_BNO055_IIC BNO;
BNO bno(&Wire, 0x28);
DFRobot_GNSS_I2C gnss(&Wire, 0x20);

FirebaseData   fbdo;
FirebaseAuth   auth;
FirebaseConfig config;

// ============================================================
//  STRUKTUR PAKET DATA
// ============================================================
typedef struct {
  BNO::sAxisAnalog_t accel, gyro, mag;
  BNO::sEulAnalog_t  euler;        // Heading SUDAH dikoreksi mirror + zero-offset
  BNO::sEulAnalog_t  euler_raw;    // Nilai mentah asli BNO055 (sebelum koreksi apapun)
  double             lat, lon, alt;
  uint8_t            sat;
  bool               imu_calibrated;
  unsigned long long timestamp_ms;
  char               datetime[32];
} UploadPacket;

// ============================================================
//  VARIABEL GLOBAL
// ============================================================
QueueHandle_t     uploadQueue;
SemaphoreHandle_t dataMutex;
volatile bool     ntpSynced     = false;
volatile bool     wifiConnected = false;

// State kalibrasi zero-offset
typedef enum {
  CALIB_WAITING,      // Menunggu GPS fix
  CALIB_STABILIZING,  // Mengumpulkan sampel
  CALIB_DONE          // Offset sudah terkunci
} CalibState;

CalibState calib_state  = CALIB_WAITING;
uint8_t    stable_count = 0;
uint32_t   gps_lost_time = 0;
float      accum_head = 0, accum_pitch = 0, accum_roll = 0;
float      offset_head = 0, offset_pitch = 0, offset_roll = 0;

// ============================================================
//  HELPER: NORMALISASI SUDUT
// ============================================================

// Heading → [0, 360)
float normalizeAngle360(float a) {
  while (a >= 360.0f) a -= 360.0f;
  while (a <    0.0f) a += 360.0f;
  return a;
}

// Pitch & Roll → [-180, +180]
float normalizeAngle180(float a) {
  while (a >  180.0f) a -= 360.0f;
  while (a < -180.0f) a += 360.0f;
  return a;
}

// ============================================================
//  KOREKSI HEADING MIRROR
//
//  Masalah:  BNO055 berputar CCW (counter-clockwise)
//            → Timur (90°) terbaca Barat (270°)
//
//  Solusi:   heading_fix = 360° − heading_raw
//            → membalik arah rotasi jadi CW (searah kompas)
//
//  Contoh hasil:
//    Utara   0° → 360° (= 0°) ✅
//    Timur  90° → 270°  ❌  tanpa fix
//    Timur  90° →  90°  ✅  dengan fix (360-270=90)
//    Selatan 180° → 180° ✅
//    Barat  270° →  90°  ❌  tanpa fix
//    Barat  270° → 270°  ✅  dengan fix (360-90=270)
// ============================================================
float fixHeadingMirror(float rawHead) {
#if HEADING_MIRROR_FIX
  return normalizeAngle360(360.0f - rawHead);
#else
  return rawHead;
#endif
}

// ============================================================
//  MESIN KALIBRASI ZERO-OFFSET
//  Dipanggil SETELAH heading sudah dikoreksi mirror
// ============================================================
void updateCalibration(bool gpsValid, const BNO::sEulAnalog_t& euler_fixed) {

  if (!gpsValid) {
    if (gps_lost_time == 0) gps_lost_time = millis();

    // Jika GPS hilang terlalu lama dan kalibrasi sudah selesai → reset
    if (calib_state == CALIB_DONE) {
      if ((millis() - gps_lost_time) > (uint32_t)GPS_LOSS_RESET_SEC * 1000UL) {
        Serial.println("[Calib] ⚠️  GPS hilang > " + String(GPS_LOSS_RESET_SEC) +
                       "s → Offset di-reset.");
        calib_state  = CALIB_WAITING;
        stable_count = 0;
        accum_head = accum_pitch = accum_roll = 0;
        offset_head = offset_pitch = offset_roll = 0;
      }
    } else {
      stable_count = 0;
      accum_head = accum_pitch = accum_roll = 0;
    }
    return;
  }

  gps_lost_time = 0;
  if (calib_state == CALIB_DONE) return;

  if (calib_state == CALIB_WAITING) {
    calib_state  = CALIB_STABILIZING;
    stable_count = 0;
    accum_head = accum_pitch = accum_roll = 0;
    Serial.println("[Calib] 📡 GPS fix! Mengumpulkan sampel...");
  }

  if (calib_state == CALIB_STABILIZING) {
    stable_count++;
    accum_head  += euler_fixed.head;
    accum_pitch += euler_fixed.pitch;
    accum_roll  += euler_fixed.roll;

    Serial.printf("[Calib] Sampel %d/%d → Head:%.1f° Pitch:%.1f° Roll:%.1f°\n",
                  stable_count, GPS_STABLE_COUNT,
                  euler_fixed.head, euler_fixed.pitch, euler_fixed.roll);

    if (stable_count >= GPS_STABLE_COUNT) {
      offset_head  = accum_head  / (float)GPS_STABLE_COUNT;
      offset_pitch = accum_pitch / (float)GPS_STABLE_COUNT;
      offset_roll  = accum_roll  / (float)GPS_STABLE_COUNT;
      calib_state  = CALIB_DONE;

      Serial.println("[Calib] ✅ Zero-offset terkunci!");
      Serial.printf("[Calib]    Offset → Head:%.2f°  Pitch:%.2f°  Roll:%.2f°\n",
                    offset_head, offset_pitch, offset_roll);
    }
  }
}

// ============================================================
//  TERAPKAN ZERO-OFFSET (nol-kan dari titik GPS fix)
// ============================================================
BNO::sEulAnalog_t applyZeroOffset(const BNO::sEulAnalog_t& euler_fixed) {
  BNO::sEulAnalog_t cal;
  cal.head  = normalizeAngle360(euler_fixed.head  - offset_head);
  cal.pitch = normalizeAngle180(euler_fixed.pitch - offset_pitch);
  cal.roll  = normalizeAngle180(euler_fixed.roll  - offset_roll);
  return cal;
}

// ============================================================
//  HELPER: STATUS BNO055
// ============================================================
void printLastOperateStatus(BNO::eStatus_t s) {
  switch (s) {
    case BNO::eStatusOK:                    Serial.println(" → OK");                        break;
    case BNO::eStatusErr:                   Serial.println(" → Error tak diketahui");        break;
    case BNO::eStatusErrDeviceNotDetect:    Serial.println(" → Perangkat tidak terdeteksi"); break;
    case BNO::eStatusErrDeviceReadyTimeOut: Serial.println(" → Device ready timeout");       break;
    case BNO::eStatusErrDeviceStatus:       Serial.println(" → Device internal error");      break;
    default:                                Serial.println(" → Status tak diketahui");       break;
  }
}

// ============================================================
//  HELPER: WAKTU
// ============================================================
unsigned long long getUnixTimestampMs() {
  struct timeval tv; gettimeofday(&tv, NULL);
  return (unsigned long long)(tv.tv_sec) * 1000ULL + (tv.tv_usec / 1000ULL);
}

void getDatetimeString(char* buf, size_t sz) {
  struct timeval tv; gettimeofday(&tv, NULL);
  struct tm tm; localtime_r(&tv.tv_sec, &tm);
  snprintf(buf, sz, "%04d-%02d-%02d %02d:%02d:%02d.%03d",
    tm.tm_year+1900, tm.tm_mon+1, tm.tm_mday,
    tm.tm_hour, tm.tm_min, tm.tm_sec, (int)(tv.tv_usec/1000));
}

bool syncNTP() {
  Serial.println("[NTP] Sinkronisasi waktu...");
  configTime(GMT_OFFSET_SEC, DST_OFFSET_SEC, NTP_SERVER);
  struct tm tm; unsigned long t = millis();
  while (!getLocalTime(&tm)) {
    if (millis()-t > NTP_TIMEOUT_MS) { Serial.println("[NTP] ⚠️ Timeout!"); return false; }
    vTaskDelay(500/portTICK_PERIOD_MS);
  }
  char buf[32]; strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
  Serial.printf("[NTP] ✅ %s WIB\n", buf);
  return true;
}

// ============================================================
//  HELPER: WIFI
// ============================================================
bool connectWiFi(unsigned long ms) {
  Serial.printf("[WiFi] Menghubungkan ke \"%s\"...\n", ssid);
  WiFi.mode(WIFI_STA); WiFi.begin(ssid, password);
  unsigned long t = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis()-t > ms) { Serial.println("\n[WiFi] ⚠️ Timeout!"); return false; }
    Serial.print("."); vTaskDelay(500/portTICK_PERIOD_MS);
  }
  Serial.printf("\n[WiFi] ✅ IP: %s\n", WiFi.localIP().toString().c_str());
  return true;
}

// ============================================================
//  CORE 0 — TASK FIREBASE
// ============================================================
void TaskFirebase(void* pvParameters) {
  UploadPacket pkt;
  while (true) {

    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("[WiFi] 🔄 Reconnect...");
      wifiConnected = false;
      WiFi.disconnect(); vTaskDelay(1000/portTICK_PERIOD_MS);
      if (connectWiFi(WIFI_RECONNECT_TIMEOUT)) {
        wifiConnected = true;
        if (!ntpSynced) ntpSynced = syncNTP();
      } else { vTaskDelay(5000/portTICK_PERIOD_MS); continue; }
    }

    if (xQueueReceive(uploadQueue, &pkt, pdMS_TO_TICKS(2000)) != pdTRUE) continue;

    if (Firebase.ready()) {
      FirebaseJson json;
      json.set("datetime",     String(pkt.datetime));
      json.set("timestamp_ms", String((uint64_t)pkt.timestamp_ms));

      json.set("lat", pkt.lat);
      json.set("lon", pkt.lon);
      json.set("alt", pkt.alt);
      json.set("sat", pkt.sat);

      json.set("imu_calibrated", pkt.imu_calibrated);

      // Heading sudah mirror-fix + zero-offset (nilai utama)
      json.set("head",  pkt.euler.head);
      json.set("pitch", pkt.euler.pitch);
      json.set("roll",  pkt.euler.roll);

      // Nilai mentah BNO055 (untuk debug)
      json.set("head_raw",  pkt.euler_raw.head);
      json.set("pitch_raw", pkt.euler_raw.pitch);
      json.set("roll_raw",  pkt.euler_raw.roll);

      json.set("ax", pkt.accel.x); json.set("ay", pkt.accel.y); json.set("az", pkt.accel.z);
      json.set("gx", pkt.gyro.x);  json.set("gy", pkt.gyro.y);  json.set("gz", pkt.gyro.z);
      json.set("mx", pkt.mag.x);   json.set("my", pkt.mag.y);   json.set("mz", pkt.mag.z);

      if (Firebase.RTDB.pushJSON(&fbdo, "/Inkubator_3T/Testing_Data", &json)) {
        Serial.printf("[Firebase] ✅ Terkirim! Key: %s\n", fbdo.pushName().c_str());
      } else {
        Serial.printf("[Firebase] ❌ Gagal: %s\n", fbdo.errorReason().c_str());
        xQueueSendToFront(uploadQueue, &pkt, 0);
        vTaskDelay(2000/portTICK_PERIOD_MS);
      }
    } else {
      xQueueSendToFront(uploadQueue, &pkt, 0);
      vTaskDelay(1000/portTICK_PERIOD_MS);
    }
  }
}

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200); delay(500);
  Serial.println("\n=========================================");
  Serial.println("  Inkubator 3T  |  v4 Heading Fix");
  Serial.println("=========================================\n");

  Wire.begin(I2C_SDA, I2C_SCL);
  dataMutex   = xSemaphoreCreateMutex();
  uploadQueue = xQueueCreate(QUEUE_SIZE, sizeof(UploadPacket));

  // BNO055
  Serial.println("--- BNO055 ---");
  bno.reset();
  for (uint8_t i = 0; bno.begin() != BNO::eStatusOK; i++) {
    Serial.printf("⚠️  Coba ke-%d", i+1);
    printLastOperateStatus(bno.lastOperateStatus);
    if (i >= 9) { Serial.println("❌ BNO055 gagal!"); break; }
    delay(2000);
  }
  if (bno.begin() == BNO::eStatusOK) Serial.println("✅ BNO055 OK!\n");

  // GNSS
  Serial.println("--- GNSS ---");
  if (!gnss.begin()) { Serial.println("⚠️  GNSS gagal!\n"); }
  else { gnss.enablePower(); Serial.println("✅ GNSS OK!\n"); }

  // WiFi
  Serial.println("--- WiFi ---");
  wifiConnected = connectWiFi(WIFI_CONNECT_TIMEOUT);
  if (wifiConnected) ntpSynced = syncNTP();
  else Serial.println("[WiFi] Mode offline, data di-queue.\n");

  // Firebase
  Serial.println("--- Firebase ---");
  config.api_key      = API_KEY;
  config.database_url = DATABASE_URL;
  if (Firebase.signUp(&config, &auth, "", ""))
    Serial.println("✅ Firebase Auth OK! (Anonymous)");
  else
    Serial.printf("❌ Auth Gagal: %s\n", config.signer.signupError.message.c_str());

  config.token_status_callback = tokenStatusCallback;
  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
  fbdo.setResponseSize(1024);
  Serial.println("✅ Firebase OK!\n");

  xTaskCreatePinnedToCore(TaskFirebase, "FirebaseTask", 10000, NULL, 1, NULL, 0);

#if HEADING_MIRROR_FIX
  Serial.println("[Calib] Heading mirror fix: AKTIF (360 - heading)");
#else
  Serial.println("[Calib] Heading mirror fix: NONAKTIF");
#endif
  Serial.println("[Calib] Menunggu GPS fix untuk zero-offset IMU...\n");
}

// ============================================================
//  CORE 1 — MAIN LOOP
// ============================================================
void loop() {
  static unsigned long lastCollect = 0;

  if (millis() - lastCollect >= SENSOR_INTERVAL_MS) {
    lastCollect = millis();

    // Baca BNO055
    BNO::sAxisAnalog_t raw_accel = bno.getAxis(BNO::eAxisAcc);
    BNO::sAxisAnalog_t raw_gyro  = bno.getAxis(BNO::eAxisGyr);
    BNO::sAxisAnalog_t raw_mag   = bno.getAxis(BNO::eAxisMag);
    BNO::sEulAnalog_t  raw_euler = bno.getEul();

    // Baca GNSS
    uint8_t current_sat = gnss.getNumSatUsed();
    double  raw_lat     = gnss.getLat().latitudeDegree;
    double  raw_lon     = gnss.getLon().lonitudeDegree;
    double  raw_alt     = gnss.getAlt();

    // Validasi GPS
    bool gpsValid = (current_sat >= 3) && (raw_lat != 0.0) &&
                    (raw_lon != 0.0)   && (raw_lon != 107.0);
    if (!gpsValid) {
      raw_lat = raw_lon = raw_alt = 0.0;
    } else {
      if (raw_lat > 0.0 && raw_lat < 10.0) raw_lat = -raw_lat;
    }

    // ── LANGKAH 1: Koreksi heading mirror ──────────────────
    //    Mengubah putaran CCW → CW agar Timur = 90°, Bukan 270°
    BNO::sEulAnalog_t euler_fixed = raw_euler;
    euler_fixed.head = fixHeadingMirror(raw_euler.head);

    // ── LANGKAH 2: Update mesin zero-offset ────────────────
    //    Menggunakan nilai yang SUDAH mirror-fix sebagai acuan
    updateCalibration(gpsValid, euler_fixed);

    // ── LANGKAH 3: Terapkan zero-offset ────────────────────
    //    Saat kalibrasi belum selesai, nilai = euler_fixed (tidak nol)
    //    Saat kalibrasi selesai, nilai = 0° di titik GPS fix
    BNO::sEulAnalog_t cal_euler = applyZeroOffset(euler_fixed);

    // Bangun paket
    UploadPacket pkt;
    pkt.accel.x = raw_accel.x * 0.00980665f;
    pkt.accel.y = raw_accel.y * 0.00980665f;
    pkt.accel.z = raw_accel.z * 0.00980665f;
    pkt.gyro.x  = raw_gyro.x  * 0.0174533f;
    pkt.gyro.y  = raw_gyro.y  * 0.0174533f;
    pkt.gyro.z  = raw_gyro.z  * 0.0174533f;
    pkt.mag      = raw_mag;
    pkt.euler    = cal_euler;   // ← Mirror fix + zero-offset
    pkt.euler_raw = raw_euler;  // ← Mentah asli BNO055
    pkt.lat      = raw_lat;
    pkt.lon      = raw_lon;
    pkt.alt      = raw_alt;
    pkt.sat      = current_sat;
    pkt.imu_calibrated = (calib_state == CALIB_DONE);
    pkt.timestamp_ms   = getUnixTimestampMs();
    getDatetimeString(pkt.datetime, sizeof(pkt.datetime));

    if (xQueueSend(uploadQueue, &pkt, 0) != pdTRUE)
      Serial.println("⚠️  Queue penuh!");

    // Log Serial
    const char* cs =
      calib_state == CALIB_DONE        ? "✅CALIB " :
      calib_state == CALIB_STABILIZING ? "⏳STABIL" : "⏸WAIT  ";

    Serial.printf(
      "[%s] %s | "
      "Head:%6.1f°(raw:%6.1f°) Pitch:%5.1f° Roll:%5.1f° | "
      "Lat:%.6f Lon:%.6f | Sat:%d\n",
      cs, pkt.datetime,
      cal_euler.head, raw_euler.head,
      cal_euler.pitch, cal_euler.roll,
      raw_lat, raw_lon, current_sat
    );
  }

  vTaskDelay(10/portTICK_PERIOD_MS);
}

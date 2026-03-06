// --- FUNGSI GET (Untuk Cek Koneksi dari Web Browser) ---
function doGet(e) {
  return ContentService.createTextOutput("✅ Script Aktif! Menerima data lengkap 9-DOF (BNO055) + GNSS dari ESP32.");
}

// --- FUNGSI POST (Menerima JSON dari ESP32) ---
function doPost(e) {
  try {
    // 1. Buka Spreadsheet berdasarkan ID (Pastikan ID ini benar)
    var ss = SpreadsheetApp.openById("spreadsheeturl");
    var sheetName = "Testing"; 
    var sheet = ss.getSheetByName(sheetName);

    // 2. Setup Header Kolom jika Sheet baru dibuat
    // Urutan header ini HARUS selaras dengan urutan data yang kita masukkan di bawah.
    if (!sheet) {
      sheet = ss.insertSheet(sheetName);
      sheet.appendRow([
        "Server Date", "Device Timestamp", 
        "Ax", "Ay", "Az", 
        "Gx", "Gy", "Gz", 
        "Mx", "My", "Mz",
        "Yaw (Heading)", "Pitch", "Roll",
        "Latitude", "Longitude", 
        "Altitude", "Satellites", "Maps Link"
      ]);
      sheet.setFrozenRows(1); // Kunci baris header
    }

    // 3. Validasi Konten POST (Mencegah error jika data kosong)
    if (!e || !e.postData || !e.postData.contents) {
      return ContentService.createTextOutput("Error: No Data Received");
    }

    // 4. Parsing Data JSON dari ESP32
    var data = JSON.parse(e.postData.contents);

    // Ambil koordinat untuk membuat tautan Google Maps yang bisa diklik
    var lat = data.lat || 0;
    var lon = data.lon || 0;
    var mapsLink = "https://www.google.com/maps?q=" + lat + "," + lon;

    // 5. Masukkan Data ke Baris Baru di Spreadsheet
    sheet.appendRow([
      new Date(),                // 1. Waktu Server Google
      data.timestamp || "",      // 2. Waktu Millis ESP32
      data.ax || 0,              // 3. Accelerometer X
      data.ay || 0,              // 4. Accelerometer Y
      data.az || 0,              // 5. Accelerometer Z
      data.gx || 0,              // 6. Gyroscope X
      data.gy || 0,              // 7. Gyroscope Y
      data.gz || 0,              // 8. Gyroscope Z
      data.mx || 0,              // 9. Magnetometer X
      data.my || 0,              // 10. Magnetometer Y
      data.mz || 0,              // 11. Magnetometer Z
      data.head || 0,            // 12. Yaw / Heading
      data.pitch || 0,           // 13. Pitch
      data.roll || 0,            // 14. Roll
      lat,                       // 15. Latitude
      lon,                       // 16. Longitude
      data.alt || 0,             // 17. Altitude
      data.sat || 0,             // 18. Jumlah Satelit
      mapsLink                   // 19. Tautan Peta
    ]);

    // 6. Memberikan balasan ke ESP32 bahwa data sukses diterima
    return ContentService.createTextOutput("SUCCESS: Full Data Logged");

  } catch (err) {
    // Tangkap error jika terjadi masalah pada script
    return ContentService.createTextOutput("ERROR Script: " + err.message);
  }
}

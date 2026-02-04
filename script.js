// --- FUNGSI GET (Untuk Cek di Browser) ---
function doGet(e) {
  return ContentService.createTextOutput("✅ Script Berjalan! Alat siap mengirim data via POST.");
}

// --- FUNGSI POST (Untuk Menerima Data ESP32) ---
function doPost(e) {
  try {
    var ss = SpreadsheetApp.openById("idspreadsheet");
    var sheetName = "Testing"; 
    var sheet = ss.getSheetByName(sheetName);

    // Buat Sheet & Header jika belum ada
    if (!sheet) {
      sheet = ss.insertSheet(sheetName);
      sheet.appendRow([
        "Server Date", "Device Timestamp", 
        "Accel X", "Accel Y", "Accel Z", 
        "Gyro X", "Gyro Y", "Gyro Z", 
        "Latitude", "Longitude", "Altitude", "Satellites", "Maps Link"
      ]);
    }

    if (!e || !e.postData || !e.postData.contents) {
      return ContentService.createTextOutput("Error: Kosong");
    }

    var data = JSON.parse(e.postData.contents);

    // Format Link Maps
    var lat = data.lat || 0;
    var lon = data.lon || 0;
    var mapsLink = "https://www.google.com/maps/search/?api=1&query=" + lat + "," + lon;

    sheet.appendRow([
      new Date(),
      data.timestamp || "",
      data.accel_x || 0, data.accel_y || 0, data.accel_z || 0,
      data.gyro_x || 0, data.gyro_y || 0, data.gyro_z || 0,
      lat, 
      lon, 
      data.alt || 0, 
      data.sat || 0, 
      mapsLink
    ]);

    return ContentService.createTextOutput("SUCCESS: Data Masuk");

  } catch (err) {
    return ContentService.createTextOutput("ERROR Script: " + err.message);
  }
}

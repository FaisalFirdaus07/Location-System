function doPost(e) {
  try {
    var ss = SpreadsheetApp.openById("halfofurlspreedsheet");
    var sheet = ss.getSheetByName("Testing");

    if (!sheet) {
      sheet = ss.insertSheet("Testing");
      sheet.appendRow([
        "Timestamp",
        "Accel X", "Accel Y", "Accel Z",
        "Gyro X", "Gyro Y", "Gyro Z",
        "Latitude", "Longitude", "Maps Link"
      ]);
    }

    if (!e || !e.postData || !e.postData.contents) {
      throw new Error("No POST data received");
    }

    var data = JSON.parse(e.postData.contents);

    sheet.appendRow([
      new Date(),
      data.accel_x || "",
      data.accel_y || "",
      data.accel_z || "",
      data.gyro_x || "",
      data.gyro_y || "",
      data.gyro_z || "",
      data.lat || "",
      data.lon || "",
      data.maps || ""
    ]);

    return ContentService
      .createTextOutput("SUCCESS: DATA WRITTEN")
      .setMimeType(ContentService.MimeType.TEXT);

  } catch (err) {
    return ContentService
      .createTextOutput("ERROR: " + err.message)
      .setMimeType(ContentService.MimeType.TEXT);
  }
}

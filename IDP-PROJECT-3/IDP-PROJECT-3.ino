// Updated ESP32 Hazard Detection System with BMP280 Integration

#include <DHT.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_BMP280.h>

// -------------------- STRUCTS --------------------
struct SensorData {
  float gas;
  float temp;
  float humidity;
  float light;
  float pressure;
  bool flame;
};

struct HazardScore {
  float fire_score;
  float smoke_score;
  float gas_score;
  float heat_score;
  float water_score;  // humidity-based damp/leak indicator
};

#define WINDOW_SIZE 20

struct RollingStats {
  float values[WINDOW_SIZE];
  int index = 0;
  bool filled = false;
};

// -------------------- PINS --------------------
#define MQ2_PIN     34
#define FLAME_PIN   26
#define DHT_PIN     27
#define LDR_PIN     35
#define SDA_PIN     21
#define SCL_PIN     22
#define DHTTYPE DHT22

// ADC noise floor — MQ2/LDR are 12-bit (0–4095), std < 10 is just noise
#define ADC_STD_FLOOR 10.0f
// Temp/humidity are small floats, lower floor acceptable
#define FLOAT_STD_FLOOR 0.5f

// -------------------- OBJECTS --------------------
DHT dht(DHT_PIN, DHTTYPE);
Adafruit_BMP280 bmp;

// -------------------- WIFI --------------------
const char* ssid = "OnePlus Nord CE4 Lite 5G";
const char* password = "12345678";

// Replace with your PC IP
const char* serverName = "http://10.89.93.70:5000/data";

// -------------------- GLOBALS --------------------
RollingStats gasStats;
RollingStats tempStats;
RollingStats humidityStats;
RollingStats lightStats;
RollingStats pressureStats;

// Persistent counters — require N consecutive hits before alerting
int fireCounter    = 0;
int smokeCounter   = 0;
int warningCounter = 0;
int gasCounter     = 0;
int heatCounter    = 0;
int waterCounter   = 0;  // sustained humidity spike = damp/leak

// -------------------- WIFI --------------------
void connectWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());
}

// -------------------- BUFFER UPDATE --------------------
void updateBuffer(RollingStats &rs, float val) {
  rs.values[rs.index] = val;
  rs.index = (rs.index + 1) % WINDOW_SIZE;
  if (rs.index == 0) rs.filled = true;
}

// -------------------- MEAN --------------------
float computeMean(RollingStats &rs) {
  int size = rs.filled ? WINDOW_SIZE : rs.index;
  if (size == 0) return 0;
  float sum = 0;
  for (int i = 0; i < size; i++) sum += rs.values[i];
  return sum / size;
}

// -------------------- STANDARD DEVIATION --------------------
float computeStd(RollingStats &rs, float mean) {
  int size = rs.filled ? WINDOW_SIZE : rs.index;
  if (size == 0) return ADC_STD_FLOOR;
  float variance = 0;
  for (int i = 0; i < size; i++) {
    float diff = rs.values[i] - mean;
    variance += diff * diff;
  }
  return sqrt(variance / size);
}

// -------------------- ANOMALY SCORE --------------------
// k = how many std deviations above mean = suspicious
// stdFloor = sensor-specific noise floor to avoid false triggers on stable readings
float computeAnomaly(float value, RollingStats &rs, float k, float stdFloor) {
  float mean = computeMean(rs);
  float std  = computeStd(rs, mean);
  if (std < stdFloor) std = stdFloor;
  float z = (value - mean) / std;
  if (z < k) return 0.0f;
  return min(1.0f, (z - k) / k);
}

float computeDropAnomaly(float value, RollingStats &rs, float k, float stdFloor) {
  float mean = computeMean(rs);
  float std  = computeStd(rs, mean);
  if (std < stdFloor) std = stdFloor;
  float z = (mean - value) / std;
  if (z < k) return 0.0f;
  return min(1.0f, (z - k) / k);
}

// -------------------- SENSOR READ --------------------
SensorData readSensors() {
  SensorData data;

  data.gas      = analogRead(MQ2_PIN);
  data.light    = analogRead(LDR_PIN);
  data.flame    = (digitalRead(FLAME_PIN) == LOW);

  // DHT retry logic
  for (int i = 0; i < 3; i++) {
    data.temp     = dht.readTemperature();
    data.humidity = dht.readHumidity();
    if (!isnan(data.temp) && !isnan(data.humidity)) break;
    Serial.println("DHT read failed. Retrying...");
    delay(200);
  }

  data.pressure = bmp.readPressure() / 100.0F;

  return data;
}

// -------------------- SENSOR DEBUG --------------------
void printSensorData(SensorData data) {
  Serial.println("\n========== SENSOR DATA ==========");
  Serial.print("Temperature: ");
  if (isnan(data.temp)) Serial.println("NaN ❌");
  else { Serial.print(data.temp); Serial.println(" °C"); }

  Serial.print("Humidity: ");
  if (isnan(data.humidity)) Serial.println("NaN ❌");
  else { Serial.print(data.humidity); Serial.println(" %"); }

  Serial.print("Gas: ");    Serial.println(data.gas);
  Serial.print("Light: ");  Serial.println(data.light);

  Serial.print("Pressure: ");
  Serial.print(data.pressure); Serial.println(" hPa");

  Serial.print("Flame: ");     Serial.println(data.flame ? "YES 🔥" : "NO");
  Serial.println("=================================\n");
}

// -------------------- HAZARD COMPUTATION --------------------
/*
 * FIRE: flame sensor is hard evidence (0.45 weight). Confirmed by:
 *   - Gas spike (combustion byproducts): 0.20
 *   - Temp spike: 0.20
 *   - Pressure drop (hot air rising, door/window effect): 0.10
 *   - Humidity drop (fire dries air): 0.05
 *   Light omitted from fire — fire itself raises light, making anomaly unreliable.
 *
 * SMOKE: no open flame, but gas rising + light dimming (particles scatter LDR).
 *   Humidity rise (steam/smoke carries moisture) adds confirmation.
 *   Gas: 0.40, Light_drop: 0.35, Humidity_rise: 0.25
 *   Light drop = anomaly on *inverted* reading (lower ADC = brighter for pull-up LDR,
 *   adjust sign below if your LDR wiring differs).
 *
 * GAS LEAK: strong gas anomaly, no flame, no smoke pattern.
 *   Pure gas: 0.70, pressure anomaly (ventilation change): 0.30
 *
 * WATER LEAK (humidity-based): no dedicated sensor.
 *   Sustained humidity spike above baseline = damp air, condensation, hidden leak.
 *   Gated: only fires if no flame and smoke_score < 0.4 (smoke also raises humidity,
 *   so we avoid double-classifying). Humidity: 0.75, pressure: 0.25.
 *   Requires 5 consecutive hits — humidity drifts slowly, need persistence to confirm.
 *
 * HEAT: high temp, low gas, no flame — electrical/overheating scenario.
 *   Temp: 0.70, pressure: 0.30
 */
HazardScore computeScore(SensorData data) {
  HazardScore score = {0, 0, 0, 0, 0};

  float gasAnom      = computeAnomaly(data.gas,      gasStats,      2.5f, ADC_STD_FLOOR);
  float tempAnom     = computeAnomaly(data.temp,     tempStats,     2.0f, FLOAT_STD_FLOOR);
  float humidAnom    = computeAnomaly(data.humidity, humidityStats, 2.0f, FLOAT_STD_FLOOR);
  float pressureAnom = computeAnomaly(data.pressure, pressureStats, 2.0f, FLOAT_STD_FLOOR);
  float humidDrop    = computeDropAnomaly(data.humidity, humidityStats, 2.0f, FLOAT_STD_FLOOR);
  float pressureDrop = computeDropAnomaly(data.pressure, pressureStats, 2.0f, FLOAT_STD_FLOOR);
  float lightDrop    = computeDropAnomaly(data.light,    lightStats,    2.0f, ADC_STD_FLOOR);

  // --- FIRE ---
  // Humidity and pressure anomaly here mean *drop* (fire dries + pressure change).
  // The anomaly fn measures deviation; for fire both rise and drop can trigger it — that's fine.
  score.fire_score =
    0.45f * (float)data.flame +
    0.25f * gasAnom +
    0.20f * tempAnom +
    0.05f * pressureDrop +
    0.05f * humidDrop;

  // --- SMOKE: gas rising, no open flame, light dropping (obscured) ---
  // lightDrop: smoke/particles darken the room → LDR reading drops below baseline
  if (!data.flame) {
    score.smoke_score =
      0.50f * gasAnom +
      0.35f * lightDrop +
      0.15f * humidAnom;
  }

  // --- GAS LEAK: strong gas, no flame, not a smoke pattern ---
  if (gasAnom > 0.6f && !data.flame && score.smoke_score < 0.5f) {
    score.gas_score =
      0.70f * gasAnom +
      0.30f * pressureAnom;  // ventilation shift or pressure from gas buildup
  }

  // --- WATER LEAK (humidity-based): sustained spike in humidity, no fire/smoke context
  // High humidity alone = damp environment, condensation, or hidden leak.
  // Gated: not already explained by smoke (smoke also raises humidity).
  // humidAnom captures deviation above baseline; pressure also rises in damp enclosed spaces.
  if (!data.flame && humidAnom > 0.45f && gasAnom < 0.35f && tempAnom < 0.35f && score.smoke_score < 0.35f) {
    score.water_score =
      0.90f * humidAnom +
      0.10f * pressureAnom;
  }

  // --- OVERHEATING: high temp, no gas, no flame ---
  if (tempAnom > 0.6f && gasAnom < 0.3f && !data.flame) {
    score.heat_score =
      0.70f * tempAnom +
      0.30f * pressureAnom;
  }

  return score;
}

// -------------------- DETECT HAZARD --------------------
String detectHazard(HazardScore score) {

  // FIRE — fast trigger (2 hits), high urgency
  if (score.fire_score > 0.65f) fireCounter++;
  else fireCounter = 0;

  // FIRE WARNING — slightly lower threshold, slightly more patience
  if (score.fire_score > 0.40f) warningCounter++;
  else warningCounter = 0;

  // SMOKE — 3 hits required (avoid single-reading flicker)
  if (score.smoke_score > 0.55f) smokeCounter++;
  else smokeCounter = 0;

  // GAS LEAK — 3 hits
  if (score.gas_score > 0.60f) gasCounter++;
  else gasCounter = 0;

  // WATER LEAK — 5 hits (humidity drifts slowly; need sustained anomaly to confirm)
  if (score.water_score > 0.65f) waterCounter++;
  else waterCounter = 0;

  // OVERHEATING — 3 hits
  if (score.heat_score > 0.70f) heatCounter++;
  else heatCounter = 0;

  // PRIORITY ORDER (most dangerous first)
  if (fireCounter    >= 2) return "DANGER";
  if (warningCounter >= 3) return "WARNING";
  if (smokeCounter   >= 3) return "SMOKE";
  if (gasCounter     >= 3) return "GAS_LEAK";
  if (waterCounter   >= 7) return "WATER_LEAK";
  if (heatCounter    >= 3) return "OVERHEATING";

  return "SAFE";
}

// -------------------- SEND TO SERVER --------------------
void sendToServer(SensorData data, HazardScore score, String hazard) {

  if (isnan(data.temp) || isnan(data.humidity)) {
    Serial.println("Invalid DHT values. Skipping send.");
    return;
  }

  if (WiFi.status() == WL_CONNECTED) {

    HTTPClient http;
    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");

    String json = "{";
    json += "\"temperature\":" + String(data.temp) + ",";
    json += "\"humidity\":"    + String(data.humidity) + ",";
    json += "\"gas\":"         + String(data.gas) + ",";
    json += "\"light\":"       + String(data.light) + ",";
    json += "\"pressure\":"    + String(data.pressure) + ",";
    json += "\"fire_score\":"  + String(score.fire_score, 4) + ",";
    json += "\"smoke_score\":" + String(score.smoke_score, 4) + ",";
    json += "\"gas_score\":"   + String(score.gas_score, 4) + ",";
    json += "\"water_score\":" + String(score.water_score, 4) + ",";
    json += "\"heat_score\":"  + String(score.heat_score, 4) + ",";
    json += "\"hazard\":\""    + hazard + "\"";
    json += "}";

    Serial.println("Sending JSON:");
    Serial.println(json);

    int httpCode = http.POST(json);
    Serial.print("HTTP Response Code: ");
    Serial.println(httpCode);

    http.end();
  } else {
    Serial.println("WiFi Disconnected!");
  }
}

// -------------------- SETUP --------------------
void setup() {
  Serial.begin(115200);

  pinMode(FLAME_PIN, INPUT);

  dht.begin();

  Wire.begin(SDA_PIN, SCL_PIN);

  if (!bmp.begin(0x76)) {
    Serial.println("BMP280 not found at 0x76, trying 0x77...");
    if (!bmp.begin(0x77)) {
      Serial.println("BMP280 not found! Continuing without pressure sensor.");
      delay(2000);
    }
  } else {
    Serial.println("BMP280 Initialized!");
  }

  connectWiFi();
}

// -------------------- LOOP --------------------
void loop() {

  SensorData data = readSensors();
  printSensorData(data);

  // UPDATE BUFFERS
  // LEARNING PHASE — wait until all buffers have enough samples
  bool learning = !gasStats.filled || !tempStats.filled ||
                  !humidityStats.filled || !lightStats.filled || !pressureStats.filled;
  if (learning) {
    updateBuffer(gasStats,      data.gas);
    updateBuffer(tempStats,     data.temp);
    updateBuffer(humidityStats, data.humidity);
    updateBuffer(lightStats,    data.light);
    updateBuffer(pressureStats, data.pressure);
    Serial.println("Learning environment...");
    delay(2000);
    return;
  }

  HazardScore score = computeScore(data);

  Serial.println("--- SCORES ---");
  Serial.print("Fire Score:  ");  Serial.println(score.fire_score);
  Serial.print("Smoke Score: ");  Serial.println(score.smoke_score);
  Serial.print("Gas Score:   ");  Serial.println(score.gas_score);
  Serial.print("Water Score: ");  Serial.println(score.water_score);
  Serial.print("Heat Score:  ");  Serial.println(score.heat_score);

  String hazard = detectHazard(score);
  Serial.print("Detected Hazard: ");
  Serial.println(hazard);

  sendToServer(data, score, hazard);

  updateBuffer(gasStats,      data.gas);
  updateBuffer(tempStats,     data.temp);
  updateBuffer(humidityStats, data.humidity);
  updateBuffer(lightStats,    data.light);
  updateBuffer(pressureStats, data.pressure);

  delay(2000);
}

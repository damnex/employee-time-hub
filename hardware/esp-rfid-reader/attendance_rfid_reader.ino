/*
  Attendance RFID Reader

  Supported boards:
  - ESP8266 (NodeMCU / D1 mini style boards)
  - ESP32

  What this sketch does:
  - Connects the reader to your Wi-Fi / phone hotspot
  - Connects to the app WebSocket endpoint at /ws/device
  - Reads MFRC522 RFID cards
  - Sends {"type":"rfid_detected","rfidUid":"...","scanTechnology":"HF_RFID"} to the server

  Important:
  This project's gate flow expects the browser to do face verification.
  So this hardware should send "rfid_detected", not "rfid_scan".
*/

#if defined(ESP8266)
#include <ESP8266WiFi.h>
#elif defined(ESP32)
#include <WiFi.h>
#else
#error Unsupported board. Use ESP8266 or ESP32.
#endif

#include <SPI.h>
#include <MFRC522.h>
#include <WebSocketsClient.h>

// Recommended debug path: use the laptop's Windows Mobile Hotspot so the ESP
// connects directly to the laptop running the app. Replace these with the
// hotspot name/password shown in Windows Settings > Mobile hotspot.
const char* WIFI_SSID = "YOUR-LAPTOP-HOTSPOT";
const char* WIFI_PASSWORD = "YOUR-HOTSPOT-PASSWORD";

// Update this to the laptop IPv4 address on the same network as the ESP.
// Current Wi-Fi adapter IPv4 on this machine: 10.161.159.9
const char* WS_HOST = "10.161.159.9";
const uint16_t WS_PORT = 5000;
const char* WS_PATH = "/ws/device?deviceId=GATE-TERMINAL-01&clientType=device";
const char* SCAN_TECHNOLOGY = "HF_RFID";

const unsigned long WIFI_RETRY_MS = 2000;
const unsigned long WIFI_CONNECT_TIMEOUT_MS = 20000;
const unsigned long CARD_DEBOUNCE_MS = 500;
const unsigned long RFID_LOOP_SETTLE_MS = 35;
const unsigned long WS_RECONNECT_MS = 1000;
const unsigned long WS_HEARTBEAT_INTERVAL_MS = 15000;
const unsigned long WS_HEARTBEAT_TIMEOUT_MS = 4000;
const unsigned long WS_RECOVERY_CHECK_MS = 3000;
const unsigned long RFID_HEALTHCHECK_MS = 2500;
const unsigned long RFID_REINIT_COOLDOWN_MS = 1200;
const uint8_t RFID_SERIAL_READ_RETRIES = 3;

#if defined(ESP8266)
const uint8_t SS_PIN = 2;    // D4
const uint8_t RST_PIN = 0;   // D3
#elif defined(ESP32)
const uint8_t SS_PIN = 5;
const uint8_t RST_PIN = 22;
const uint8_t SCK_PIN = 18;
const uint8_t MISO_PIN = 19;
const uint8_t MOSI_PIN = 23;
#endif

WebSocketsClient webSocket;
MFRC522 mfrc522(SS_PIN, RST_PIN);

String lastUid = "";
unsigned long lastCardSentAt = 0;
unsigned long lastWifiRetryAt = 0;
bool socketConnected = false;
bool readerInitialized = false;
bool readerReadyAnnounced = false;
byte readerVersion = 0;
unsigned long lastRfidHealthcheckAt = 0;
unsigned long lastRfidInitAt = 0;
unsigned long lastWsRecoveryCheckAt = 0;
unsigned long lastWsConnectAttemptAt = 0;
bool webSocketStarted = false;

bool isReaderVersionValid(byte version) {
  return version != 0x00 && version != 0xFF;
}

const __FlashStringHelper* wifiStatusLabel(wl_status_t status) {
  switch (status) {
    case WL_CONNECTED:
      return F("CONNECTED");
    case WL_NO_SSID_AVAIL:
      return F("SSID_NOT_FOUND");
    case WL_CONNECT_FAILED:
      return F("CONNECT_FAILED");
    case WL_CONNECTION_LOST:
      return F("CONNECTION_LOST");
    case WL_DISCONNECTED:
      return F("DISCONNECTED");
#if defined(ESP8266)
    case WL_WRONG_PASSWORD:
      return F("WRONG_PASSWORD");
    case WL_IDLE_STATUS:
      return F("IDLE");
    case WL_SCAN_COMPLETED:
      return F("SCAN_COMPLETED");
#elif defined(ESP32)
    case WL_IDLE_STATUS:
      return F("IDLE");
    case WL_SCAN_COMPLETED:
      return F("SCAN_COMPLETED");
#endif
    default:
      return F("UNKNOWN");
  }
}

void printServerEndpoint() {
  Serial.print(F("ws://"));
  Serial.print(WS_HOST);
  Serial.print(':');
  Serial.print(WS_PORT);
  Serial.println(WS_PATH);
}

void logNetworkSnapshot(const __FlashStringHelper* context) {
  Serial.print(F("[NET] "));
  Serial.println(context);
  Serial.print(F("[NET] Wi-Fi status: "));
  Serial.println(wifiStatusLabel(WiFi.status()));
  Serial.print(F("[NET] SSID: "));
  Serial.println(WIFI_SSID);
  Serial.print(F("[NET] Local IP: "));
  Serial.println(WiFi.localIP());
  Serial.print(F("[NET] Gateway: "));
  Serial.println(WiFi.gatewayIP());
  Serial.print(F("[NET] Subnet mask: "));
  Serial.println(WiFi.subnetMask());
  Serial.print(F("[NET] RSSI: "));
  Serial.print(WiFi.RSSI());
  Serial.println(F(" dBm"));
  Serial.print(F("[NET] Server endpoint: "));
  printServerEndpoint();
}

String formatHexByte(byte value) {
  String hex = String(value, HEX);
  hex.toUpperCase();
  if (hex.length() < 2) {
    hex = "0" + hex;
  }
  return hex;
}

void printDivider() {
  Serial.println(F("--------------------------------------------------"));
}

bool initializeRfidReader(bool forceLog = false) {
  const unsigned long now = millis();
  if (!forceLog && lastRfidInitAt != 0 && (now - lastRfidInitAt) < RFID_REINIT_COOLDOWN_MS) {
    return readerInitialized;
  }

  lastRfidInitAt = now;

#if defined(ESP32)
  SPI.begin(SCK_PIN, MISO_PIN, MOSI_PIN, SS_PIN);
#else
  SPI.begin();
#endif

  mfrc522.PCD_Init();
  delay(4);
  mfrc522.PCD_AntennaOn();
  delay(2);

  readerVersion = mfrc522.PCD_ReadRegister(MFRC522::VersionReg);
  readerInitialized = isReaderVersionValid(readerVersion);

  if (!readerInitialized) {
    readerReadyAnnounced = false;
    printDivider();
    Serial.println(F("[RFID] MFRC522 not responding."));
    Serial.println(F("[RFID] Check 3.3V power, GND, and SPI wiring."));
    Serial.print(F("[RFID] Version register: 0x"));
    Serial.println(formatHexByte(readerVersion));
    printDivider();
    return false;
  }

  if (forceLog) {
    printDivider();
    Serial.print(F("[RFID] MFRC522 detected. Version register: 0x"));
    Serial.println(formatHexByte(readerVersion));
    Serial.println(F("[RFID] Antenna on. Reader is ready to detect tags."));
    printDivider();
  }

  return true;
}

void ensureRfidReaderReady() {
  const unsigned long now = millis();
  if ((now - lastRfidHealthcheckAt) < RFID_HEALTHCHECK_MS) {
    return;
  }

  lastRfidHealthcheckAt = now;
  byte currentVersion = mfrc522.PCD_ReadRegister(MFRC522::VersionReg);

  if (!isReaderVersionValid(currentVersion)) {
    readerInitialized = false;
    initializeRfidReader(true);
    return;
  }

  readerVersion = currentVersion;
}

void announceReaderReadyIfAvailable() {
  if (
    readerReadyAnnounced
    || !readerInitialized
    || !socketConnected
    || WiFi.status() != WL_CONNECTED
  ) {
    return;
  }

  printDivider();
  Serial.println(F("[READY] Reader connected through IP and ready for reading."));
  Serial.print(F("[READY] Reader IP: "));
  Serial.println(WiFi.localIP());
  Serial.print(F("[READY] Server endpoint: ws://"));
  Serial.print(WS_HOST);
  Serial.print(':');
  Serial.print(WS_PORT);
  Serial.println(WS_PATH);
  Serial.println(F("[READY] Tap a badge on the reader."));
  printDivider();

  readerReadyAnnounced = true;
}

void configureWifiStation() {
  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);

#if defined(ESP8266)
  WiFi.setSleepMode(WIFI_NONE_SLEEP);
#elif defined(ESP32)
  WiFi.setSleep(false);
#endif
}

void connectToWifi() {
  Serial.printf("Connecting to Wi-Fi SSID \"%s\"", WIFI_SSID);
  configureWifiStation();
  WiFi.disconnect();
  delay(100);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long startedAt = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startedAt < WIFI_CONNECT_TIMEOUT_MS) {
    delay(500);
    Serial.print('.');
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println(F("Wi-Fi connected."));
    Serial.print(F("Device IP: "));
    Serial.println(WiFi.localIP());
    Serial.print(F("Target server: "));
    printServerEndpoint();
    Serial.print(F("Gateway: "));
    Serial.println(WiFi.gatewayIP());
    Serial.print(F("Signal strength: "));
    Serial.print(WiFi.RSSI());
    Serial.println(F(" dBm"));
  } else {
    Serial.println(F("Wi-Fi connect timeout. Will retry automatically."));
    Serial.print(F("Wi-Fi status after timeout: "));
    Serial.println(wifiStatusLabel(WiFi.status()));
    readerReadyAnnounced = false;
  }
}

void webSocketEvent(WStype_t type, uint8_t* payload, size_t length) {
  switch (type) {
    case WStype_DISCONNECTED:
      socketConnected = false;
      readerReadyAnnounced = false;
      Serial.println(F("[WS] Disconnected from server."));
      logNetworkSnapshot(F("Connection closed before handshake or after a drop."));
      Serial.println(F("[WS] If this repeats, check Windows Firewall on TCP 5000 and verify both devices are on the same hotspot."));
      break;

    case WStype_CONNECTED:
      socketConnected = true;
      lastWsConnectAttemptAt = millis();
      Serial.print(F("[WS] Connected to: "));
      Serial.println(reinterpret_cast<const char*>(payload));
      announceReaderReadyIfAvailable();
      break;

    case WStype_TEXT:
      Serial.print(F("[WS] Message: "));
      for (size_t i = 0; i < length; i++) {
        Serial.print(static_cast<char>(payload[i]));
      }
      Serial.println();
      break;

    case WStype_ERROR:
      socketConnected = false;
      readerReadyAnnounced = false;
      Serial.println(F("[WS] Error event received."));
      if (payload != nullptr && length > 0) {
        Serial.print(F("[WS] Error payload: "));
        for (size_t i = 0; i < length; i++) {
          Serial.print(static_cast<char>(payload[i]));
        }
        Serial.println();
      } else {
        Serial.println(F("[WS] No error payload provided by the library."));
      }
      logNetworkSnapshot(F("WebSocket error details"));
      break;

    case WStype_PING:
      Serial.println(F("[WS] Ping received from server."));
      break;

    case WStype_PONG:
      Serial.println(F("[WS] Pong received from server."));
      break;

    default:
      break;
  }
}

void connectWebSocket() {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }

  if (webSocketStarted) {
    webSocket.disconnect();
    delay(20);
  }

  Serial.print(F("[WS] Starting WebSocket client -> "));
  printServerEndpoint();
  webSocket.begin(WS_HOST, WS_PORT, WS_PATH);
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(WS_RECONNECT_MS);
  webSocket.enableHeartbeat(WS_HEARTBEAT_INTERVAL_MS, WS_HEARTBEAT_TIMEOUT_MS, 2);
  lastWsConnectAttemptAt = millis();
  webSocketStarted = true;
}

void ensureWebSocketConnected() {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }

  if (socketConnected) {
    return;
  }

  const unsigned long now = millis();
  if ((now - lastWsRecoveryCheckAt) < WS_RECOVERY_CHECK_MS) {
    return;
  }

  lastWsRecoveryCheckAt = now;
  if (webSocketStarted && (now - lastWsConnectAttemptAt) < WS_RECOVERY_CHECK_MS) {
    return;
  }

  Serial.println(F("[WS] Socket offline. Restarting WebSocket client."));
  connectWebSocket();
}

String readCardUid() {
  if (!readerInitialized) {
    return "";
  }

  if (!mfrc522.PICC_IsNewCardPresent()) {
    return "";
  }

  bool uidRead = false;
  for (uint8_t attempt = 0; attempt < RFID_SERIAL_READ_RETRIES; attempt++) {
    if (mfrc522.PICC_ReadCardSerial()) {
      uidRead = true;
      break;
    }
    delay(5);
  }

  if (!uidRead) {
    Serial.println(F("[RFID] Card detected but UID read failed. Reinitializing reader."));
    readerInitialized = false;
    initializeRfidReader(true);
    return "";
  }

  String uid = "";
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    if (mfrc522.uid.uidByte[i] < 0x10) {
      uid += '0';
    }
    uid += String(mfrc522.uid.uidByte[i], HEX);
  }

  uid.toUpperCase();
  mfrc522.PICC_HaltA();
  mfrc522.PCD_StopCrypto1();
  return uid;
}

void sendRfidDetected(const String& uid) {
  if (!socketConnected) {
    Serial.println(F("[RFID] Card read, but WebSocket is offline."));
    return;
  }

  String payload =
    "{\"type\":\"rfid_detected\",\"rfidUid\":\"" + uid
    + "\",\"scanTechnology\":\"" + String(SCAN_TECHNOLOGY) + "\"}";

  Serial.print(F("[RFID] Sending UID: "));
  Serial.println(uid);
  webSocket.sendTXT(payload);
}

void ensureWifi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  readerReadyAnnounced = false;
  socketConnected = false;

  if (millis() - lastWifiRetryAt < WIFI_RETRY_MS) {
    return;
  }

  lastWifiRetryAt = millis();
  Serial.println(F("[Wi-Fi] Connection lost. Retrying..."));
  Serial.print(F("[Wi-Fi] Current status: "));
  Serial.println(wifiStatusLabel(WiFi.status()));
  WiFi.disconnect();
  connectToWifi();
  if (WiFi.status() == WL_CONNECTED) {
    connectWebSocket();
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  printDivider();
  Serial.println(F("Attendance RFID Reader starting..."));
  Serial.println(F("Recommended debug setup: enable Windows Mobile Hotspot on the laptop."));
  Serial.println(F("Update WIFI_SSID/WIFI_PASSWORD to the hotspot name and password."));
  Serial.println(F("Set WS_HOST to the laptop hotspot IPv4 (commonly 192.168.137.1)."));
  printDivider();

  connectToWifi();
  connectWebSocket();
  initializeRfidReader(true);
  Serial.println(F("MFRC522 initialized. Tap a badge on the reader."));
  announceReaderReadyIfAvailable();
  printDivider();
}

void loop() {
  ensureWifi();
  webSocket.loop();
  ensureWebSocketConnected();
  ensureRfidReaderReady();
  announceReaderReadyIfAvailable();

  const String uid = readCardUid();
  if (uid.isEmpty()) {
    return;
  }

  const unsigned long now = millis();
  if (uid == lastUid && (now - lastCardSentAt) < CARD_DEBOUNCE_MS) {
    Serial.print(F("[RFID] Duplicate tap ignored: "));
    Serial.println(uid);
    return;
  }

  lastUid = uid;
  lastCardSentAt = now;
  sendRfidDetected(uid);
  delay(RFID_LOOP_SETTLE_MS);
}


/*
  Attendance RFID Reader

  Supported boards:
  - ESP8266 (NodeMCU / D1 mini style boards)
  - ESP32

  What this sketch does:
  - Connects the reader to your Wi-Fi / phone hotspot
  - Connects to the app WebSocket endpoint at /ws/device
  - Reads MFRC522 RFID cards
  - Sends {"type":"rfid_detected","rfidUid":"..."} to the server

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

// Hotspot credentials for the presentation setup.
const char* WIFI_SSID = "One Plus";
const char* WIFI_PASSWORD = "20022007";

// Replace this with the laptop IPv4 address after the laptop joins the hotspot.
// Example: 192.168.191.120
const char* WS_HOST = "192.168.0.100";
const uint16_t WS_PORT = 5000;
const char* WS_PATH = "/ws/device?deviceId=GATE-TERMINAL-01&clientType=device";

const unsigned long WIFI_RETRY_MS = 5000;
const unsigned long CARD_DEBOUNCE_MS = 1500;

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

void printDivider() {
  Serial.println(F("--------------------------------------------------"));
}

void connectToWifi() {
  Serial.printf("Connecting to Wi-Fi SSID \"%s\"", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long startedAt = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startedAt < 20000) {
    delay(500);
    Serial.print('.');
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println(F("Wi-Fi connected."));
    Serial.print(F("Device IP: "));
    Serial.println(WiFi.localIP());
    Serial.print(F("Target server: ws://"));
    Serial.print(WS_HOST);
    Serial.print(':');
    Serial.print(WS_PORT);
    Serial.println(WS_PATH);
  } else {
    Serial.println(F("Wi-Fi connect timeout. Will retry automatically."));
  }
}

void webSocketEvent(WStype_t type, uint8_t* payload, size_t length) {
  switch (type) {
    case WStype_DISCONNECTED:
      socketConnected = false;
      Serial.println(F("[WS] Disconnected from server."));
      break;

    case WStype_CONNECTED:
      socketConnected = true;
      Serial.print(F("[WS] Connected to: "));
      Serial.println(reinterpret_cast<const char*>(payload));
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
      Serial.println(F("[WS] Error."));
      break;

    default:
      break;
  }
}

void connectWebSocket() {
  webSocket.begin(WS_HOST, WS_PORT, WS_PATH);
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(3000);
  webSocket.enableHeartbeat(15000, 3000, 2);
}

String readCardUid() {
  if (!mfrc522.PICC_IsNewCardPresent()) {
    return "";
  }

  if (!mfrc522.PICC_ReadCardSerial()) {
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

  const String payload =
    "{\"type\":\"rfid_detected\",\"rfidUid\":\"" + uid + "\"}";

  Serial.print(F("[RFID] Sending UID: "));
  Serial.println(uid);
  webSocket.sendTXT(payload);
}

void ensureWifi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  if (millis() - lastWifiRetryAt < WIFI_RETRY_MS) {
    return;
  }

  lastWifiRetryAt = millis();
  Serial.println(F("[Wi-Fi] Connection lost. Retrying..."));
  WiFi.disconnect();
  connectToWifi();
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  printDivider();
  Serial.println(F("Attendance RFID Reader starting..."));
  Serial.println(F("Before use, set WS_HOST to the laptop hotspot IPv4 address."));
  printDivider();

  connectToWifi();
  connectWebSocket();

#if defined(ESP32)
  SPI.begin(SCK_PIN, MISO_PIN, MOSI_PIN, SS_PIN);
#else
  SPI.begin();
#endif

  mfrc522.PCD_Init();
  delay(4);

  Serial.println(F("MFRC522 initialized. Tap a badge on the reader."));
  printDivider();
}

void loop() {
  ensureWifi();
  webSocket.loop();

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
  delay(250);
}

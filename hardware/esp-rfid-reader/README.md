# ESP RFID Reader

This firmware matches the WebSocket device protocol used by this app.

File:
- `hardware/esp-rfid-reader/attendance_rfid_reader.ino`

What to change before the presentation:
- In `attendance_rfid_reader.ino`, set `WS_HOST` to the laptop IPv4 address after the laptop connects to the `One Plus` hotspot.
- Keep `WS_PORT` as `5000`.
- Keep `WS_PATH` as `/ws/device?deviceId=GATE-TERMINAL-01&clientType=device`.

How to get the laptop IP:
1. Connect the laptop to the `One Plus` hotspot.
2. Start this app.
3. Run `ipconfig` on the laptop.
4. Copy the Wi-Fi IPv4 address into `WS_HOST`.
5. Upload the sketch to the board.

Arduino libraries needed:
- `MFRC522`
- `WebSockets` by Markus Sattler

Serial monitor troubleshooting:
- If you see `[RFID] MFRC522 not responding` with version `0x00` or `0xFF`, the reader module is not wired or powered correctly.
- If you see `[RFID] Card detected but UID read failed`, the reader saw RF activity but could not read the tag cleanly. Re-seat wiring, shorten jumper wires, and keep the module on stable `3.3V`.
- If you see the ready banner but no UID lines when tapping, verify the badge is a supported 13.56 MHz tag for MFRC522.

Wiring for ESP8266 (NodeMCU style):
- `SDA / SS` -> `D4` / GPIO2
- `RST` -> `D3` / GPIO0
- `SCK` -> `D5` / GPIO14
- `MISO` -> `D6` / GPIO12
- `MOSI` -> `D7` / GPIO13
- `3.3V` -> `3.3V`
- `GND` -> `GND`

Wiring for ESP32:
- `SDA / SS` -> GPIO5
- `RST` -> GPIO22
- `SCK` -> GPIO18
- `MISO` -> GPIO19
- `MOSI` -> GPIO23
- `3.3V` -> `3.3V`
- `GND` -> `GND`

Why the sketch sends `rfid_detected`:
- The gate page in this app captures the live face from the browser camera.
- If the hardware sends `rfid_scan` without face data, the server will reject it.
- `rfid_detected` is the correct event for both enrollment and gate tap detection in this project.

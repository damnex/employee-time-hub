# RFID Service

Standalone FastAPI service for a `UHFReader18` serial reader.

## Features

- Dedicated binary parser module for CRC validation, packet framing, and EPC extraction.
- Binary packet parsing with CRC-16 validation.
- EPC extraction for spontaneous `0xEE` frames and inventory responses.
- Background reader thread with reconnection.
- Rolling buffer management for fragmented, concatenated, and noisy serial traffic.
- ENTRY / EXIT event processing using `last_seen` and `active_tags`.
- Registration mode that requires one stable tag repeated at least 7 times.
- Reader power control and work mode control.

## Run

```bash
python -m pip install -r rfid_service/requirements.txt
python -m rfid_service.main
```

The service listens on `http://127.0.0.1:8001` by default.

## Project Layout

```text
rfid_service/
├── api.py
├── controller.py
├── main.py
├── parser.py
├── processor.py
└── reader.py
```

## API

- `POST /connect`
- `POST /disconnect`
- `POST /start`
- `POST /stop`
- `POST /set-power`
- `POST /set-mode`
- `POST /set-transport-mode`
- `POST /set-buzzer`
- `GET /tags`
- `GET /active-tags`
- `GET /registration-tag`
- `GET /status`

## Notes

- `normal` mode applies scan mode + inventory multiple + power `30`.
- `registration` mode applies scan mode + inventory single + power `1`.
- `scan` transport uses the reader's continuous `0xEE` output stream.
- `answer` transport keeps the reader in host-driven answer mode and polls with `0x01` or `0x0F`.
- The buzzer toggle maps to the UHFReader18 work-mode `Mode_state` bit 2 (`0 = on`, `1 = off`).
- `POST /connect` opens the serial connection first. `POST /start` begins continuous UHF processing.
- If you need raw packet logging, pass `debug_raw: true` in `POST /connect` or `POST /start`.

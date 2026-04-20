from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .controller import RFIDController


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(title="UHF RFID Reader Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

controller = RFIDController()


class ReaderConnectionRequest(BaseModel):
    port: str = Field(default="COM3", min_length=1)
    baudrate: int = Field(default=57600, ge=1200, le=921600)
    debug_raw: bool = False


class SetPowerRequest(BaseModel):
    level: int = Field(ge=0, le=30)


class SetModeRequest(BaseModel):
    mode: str = Field(min_length=1)


def _service_error(exc: Exception) -> HTTPException:
    detail = str(exc) or exc.__class__.__name__
    return HTTPException(status_code=400, detail=detail)


@app.post("/connect")
def connect_reader(payload: ReaderConnectionRequest) -> dict[str, object]:
    try:
        return controller.connect(port=payload.port, baudrate=payload.baudrate, debug_raw=payload.debug_raw)
    except Exception as exc:  # noqa: BLE001
        raise _service_error(exc) from exc


@app.post("/disconnect")
def disconnect_reader() -> dict[str, object]:
    try:
        return controller.disconnect()
    except Exception as exc:  # noqa: BLE001
        raise _service_error(exc) from exc


@app.post("/start")
def start_reader(payload: ReaderConnectionRequest) -> dict[str, object]:
    try:
        return controller.start(port=payload.port, baudrate=payload.baudrate, debug_raw=payload.debug_raw)
    except Exception as exc:  # noqa: BLE001
        raise _service_error(exc) from exc


@app.post("/stop")
def stop_reader() -> dict[str, object]:
    try:
        return controller.stop()
    except Exception as exc:  # noqa: BLE001
        raise _service_error(exc) from exc


@app.post("/set-power")
def set_power(payload: SetPowerRequest) -> dict[str, object]:
    try:
        return controller.set_power(payload.level)
    except Exception as exc:  # noqa: BLE001
        raise _service_error(exc) from exc


@app.post("/set-mode")
def set_mode(payload: SetModeRequest) -> dict[str, object]:
    try:
        return controller.set_mode(payload.mode)
    except Exception as exc:  # noqa: BLE001
        raise _service_error(exc) from exc


@app.get("/tags")
def get_tags() -> dict[str, object]:
    return controller.get_tags()


@app.get("/active-tags")
def get_active_tags() -> dict[str, object]:
    return controller.get_active_tags()


@app.get("/registration-tag")
def get_registration_tag() -> dict[str, object]:
    return controller.get_registration_tag()


@app.get("/status")
def get_status() -> dict[str, object]:
    return controller.get_status()

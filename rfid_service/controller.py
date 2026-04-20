from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass

from .processor import TagProcessor
from .reader import ReaderConfig, ReaderInfo, SerialRFIDReader, WorkModeConfig


LOGGER = logging.getLogger("rfid_service.controller")

MODE_DEFAULT_POWER = {
    "normal": 30,
    "registration": 8,
}


@dataclass(slots=True)
class ControllerState:
    port: str = "COM3"
    baudrate: int = 57600
    connected: bool = False
    running: bool = False
    current_mode: str = "normal"
    current_power: int = 30
    debug_raw: bool = False
    last_error: str | None = None


class RFIDController:
    def __init__(self) -> None:
        self._state = ControllerState()
        self._processor = TagProcessor()
        self._reader: SerialRFIDReader | None = None
        self._reader_info: ReaderInfo | None = None
        self._lock = threading.RLock()
        self._needs_runtime_sync = False

    def connect(
        self,
        *,
        port: str | None = None,
        baudrate: int | None = None,
        debug_raw: bool | None = None,
    ) -> dict[str, object]:
        with self._lock:
            desired_port = port or self._state.port
            desired_baudrate = baudrate or self._state.baudrate
            desired_debug = self._state.debug_raw if debug_raw is None else debug_raw

            if self._reader is not None:
                if (
                    desired_port == self._state.port
                    and desired_baudrate == self._state.baudrate
                    and desired_debug == self._state.debug_raw
                ):
                    return self.get_status()
                self.disconnect()

            self._state.port = desired_port
            self._state.baudrate = desired_baudrate
            self._state.debug_raw = desired_debug
            self._state.last_error = None

            config = ReaderConfig(
                port=desired_port,
                baudrate=desired_baudrate,
                debug_raw=desired_debug,
            )
            reader = SerialRFIDReader(
                config=config,
                on_tags=self._handle_detected_tags,
                on_connection_change=self._handle_connection_change,
            )
            try:
                reader.start()
                self._reader = reader
                self._processor.set_mode(self._state.current_mode)
                self._state.running = False
                self._state.connected = reader.connected
                self._needs_runtime_sync = True

                try:
                    self._reader_info = reader.get_reader_info()
                    self._state.current_power = self._reader_info.power
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Unable to read reader info during startup: %s", exc)

                try:
                    work_mode = reader.get_work_mode()
                    self._state.current_mode = self._mode_from_work_mode(work_mode)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Unable to read work mode during startup: %s", exc)

                self._sync_runtime_state_locked()
                return self.get_status()
            except Exception:
                reader.stop()
                self._reader = None
                self._reader_info = None
                self._state.connected = False
                self._state.running = False
                self._needs_runtime_sync = False
                raise

    def start(
        self,
        *,
        port: str | None = None,
        baudrate: int | None = None,
        debug_raw: bool | None = None,
    ) -> dict[str, object]:
        with self._lock:
            if self._reader is None or not self._state.connected:
                self.connect(port=port, baudrate=baudrate, debug_raw=debug_raw)

            self._sync_runtime_state_locked()
            if self._state.running:
                return self.get_status()

            self._processor.stop(force_exit=True)
            self._processor.set_mode(self._state.current_mode)
            self._processor.start()
            self._state.running = True
            LOGGER.info("Reader stream started in %s mode.", self._state.current_mode)
            return self.get_status()

    def stop(self) -> dict[str, object]:
        with self._lock:
            if not self._state.running:
                return self.get_status()

            self._processor.stop(force_exit=True)
            self._state.running = False
            LOGGER.info("Reader stream stopped.")
            return self.get_status()

    def disconnect(self) -> dict[str, object]:
        with self._lock:
            if self._reader is not None:
                self._reader.stop()
            self._reader = None
            self._reader_info = None
            self._processor.stop(force_exit=True)
            self._state.connected = False
            self._state.running = False
            self._needs_runtime_sync = False
            return self.get_status()

    def set_power(self, level: int) -> dict[str, object]:
        with self._lock:
            self._sync_runtime_state_locked()
            reader = self._require_reader()
            reader.set_power(level)
            self._state.current_power = level
            LOGGER.info("Power changed to %s", level)
            return self.get_status()

    def set_mode(self, mode: str) -> dict[str, object]:
        normalized_mode = self._normalize_mode(mode)
        with self._lock:
            self._sync_runtime_state_locked()
            reader = self._require_reader()
            reader.set_work_mode(self._work_mode_for(normalized_mode))
            self._processor.set_mode(normalized_mode)
            self._state.current_mode = normalized_mode
            LOGGER.info("Mode changed to %s", normalized_mode)

            default_power = MODE_DEFAULT_POWER[normalized_mode]
            reader.set_power(default_power)
            self._state.current_power = default_power
            LOGGER.info("Power changed to %s", default_power)
            return self.get_status()

    def get_tags(self) -> dict[str, object]:
        with self._lock:
            self._sync_runtime_state_locked()
            return {
                **self.get_status(),
                **self._processor.snapshot(),
            }

    def get_active_tags(self) -> dict[str, object]:
        with self._lock:
            self._sync_runtime_state_locked()
            return {
                **self.get_status(),
                "active_tags": self._processor.get_active_tags(),
            }

    def get_registration_tag(self) -> dict[str, object]:
        with self._lock:
            self._sync_runtime_state_locked()
            registration = self._processor.get_registration_state()
            return {
                **self.get_status(),
                "registration": registration,
                "selected_tag": registration.get("selected_tag"),
            }

    def get_status(self) -> dict[str, object]:
        with self._lock:
            self._sync_runtime_state_locked()
            return {
                **asdict(self._state),
                "reader_info": asdict(self._reader_info) if self._reader_info is not None else None,
            }

    def _require_reader(self) -> SerialRFIDReader:
        if self._reader is None or not self._state.connected:
            raise RuntimeError("RFID reader is not connected.")
        return self._reader

    def _handle_detected_tags(self, packet, tags: list[str]) -> None:
        if not self._state.running:
            return
        self._processor.process_tags(tags, raw_hex=packet.raw_hex)

    def _handle_connection_change(self, connected: bool) -> None:
        with self._lock:
            self._state.connected = connected
            if connected:
                self._state.last_error = None
                self._needs_runtime_sync = True
            else:
                self._state.last_error = f"RFID reader on {self._state.port} disconnected."

    def _normalize_mode(self, mode: str) -> str:
        normalized = mode.strip().lower()
        if normalized not in MODE_DEFAULT_POWER:
            raise ValueError("Mode must be one of: normal, registration.")
        return normalized

    def _work_mode_for(self, mode: str) -> WorkModeConfig:
        if mode == "registration":
            return WorkModeConfig.registration_scan()
        return WorkModeConfig.normal_scan()

    def _mode_from_work_mode(self, mode: WorkModeConfig) -> str:
        if mode.read_mode == 0x01 and mode.mem_inven == 0x05:
            return "registration"
        return "normal"

    def _apply_runtime_configuration_locked(self) -> None:
        reader = self._require_reader()
        reader.set_work_mode(self._work_mode_for(self._state.current_mode))
        self._processor.set_mode(self._state.current_mode)
        reader.set_power(self._state.current_power)
        self._needs_runtime_sync = False
        LOGGER.info(
            "Reader runtime synchronized: mode=%s power=%s",
            self._state.current_mode,
            self._state.current_power,
        )

    def _sync_runtime_state_locked(self) -> None:
        if not self._needs_runtime_sync or not self._state.connected:
            return
        self._apply_runtime_configuration_locked()

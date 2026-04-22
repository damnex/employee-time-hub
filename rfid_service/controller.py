from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass

from .processor import TagProcessor
from .parser import extract_epcs_from_packet
from .reader import (
    ReaderConfig,
    ReaderInfo,
    SerialPortInfo,
    SerialRFIDReader,
    WorkModeConfig,
    detect_reader_port,
    list_serial_ports,
)


LOGGER = logging.getLogger("rfid_service.controller")

MODE_DEFAULT_POWER = {
    "normal": 30,
    "registration": 1,
}
ANSWER_MODE_RETRY_SECONDS = 0.2
ANSWER_MODE_IDLE_SECONDS = 0.05


@dataclass(slots=True)
class ControllerState:
    port: str = "COM3"
    baudrate: int = 57600
    connected: bool = False
    running: bool = False
    current_mode: str = "normal"
    transport_mode: str = "scan"
    buzzer_enabled: bool = False
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
        self._supports_work_mode_readback = True
        self._answer_poll_thread: threading.Thread | None = None
        self._answer_poll_stop_event: threading.Event | None = None
        self._answer_poll_generation = 0

    def connect(
        self,
        *,
        port: str | None = None,
        baudrate: int | None = None,
        debug_raw: bool | None = None,
    ) -> dict[str, object]:
        with self._lock:
            desired_port = self._normalize_port_name(port or self._state.port)
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

            try:
                return self._connect_locked(
                    port=desired_port,
                    baudrate=desired_baudrate,
                    debug_raw=desired_debug,
                )
            except Exception as primary_exc:
                LOGGER.warning("RFID connection failed on %s: %s", desired_port, primary_exc)
                auto_detected = self._auto_detect_port_locked(
                    preferred_port=desired_port,
                    baudrate=desired_baudrate,
                    debug_raw=desired_debug,
                )
                if auto_detected is not None and auto_detected[0].device != desired_port:
                    fallback_port = auto_detected[0].device
                    LOGGER.info(
                        "Auto-detected RFID reader on %s after %s failed.",
                        fallback_port,
                        desired_port,
                    )
                    self._state.port = fallback_port
                    return self._connect_locked(
                        port=fallback_port,
                        baudrate=desired_baudrate,
                        debug_raw=desired_debug,
                    )

                self._state.last_error = str(primary_exc)
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
            self._ensure_answer_polling_locked()
            LOGGER.info(
                "Reader stream started in %s mode using %s transport.",
                self._state.current_mode,
                self._state.transport_mode,
            )
            return self.get_status()

    def stop(self) -> dict[str, object]:
        with self._lock:
            if not self._state.running:
                return self.get_status()

            self._processor.stop(force_exit=True)
            self._stop_answer_polling_locked()
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
            self._stop_answer_polling_locked()
            self._state.connected = False
            self._state.running = False
            self._needs_runtime_sync = False
            self._supports_work_mode_readback = True
            return self.get_status()

    def detect_port(
        self,
        *,
        baudrate: int | None = None,
        debug_raw: bool | None = None,
    ) -> dict[str, object]:
        with self._lock:
            desired_baudrate = baudrate or self._state.baudrate
            desired_debug = self._state.debug_raw if debug_raw is None else debug_raw

            if self._state.connected:
                active_port = SerialPortInfo(
                    device=self._state.port,
                    description="Active RFID reader connection",
                )
                return {
                    **self.get_status(),
                    "detected_port": asdict(active_port),
                    "detected_reader_info": asdict(self._reader_info) if self._reader_info is not None else None,
                }

            detected = self._auto_detect_port_locked(
                preferred_port=self._state.port,
                baudrate=desired_baudrate,
                debug_raw=desired_debug,
            )
            if detected is None:
                available_ports = list_serial_ports()
                if not available_ports:
                    self._state.last_error = "No serial ports found for RFID auto-detect."
                    raise RuntimeError(self._state.last_error)

                available_names = ", ".join(port.device for port in available_ports)
                self._state.last_error = (
                    f"No UHF reader detected on available serial ports: {available_names}."
                )
                raise RuntimeError(self._state.last_error)

            detected_port, detected_reader_info = detected
            self._state.port = detected_port.device
            self._state.baudrate = desired_baudrate
            self._state.debug_raw = desired_debug
            self._state.last_error = None
            return {
                **self.get_status(),
                "detected_port": asdict(detected_port),
                "detected_reader_info": asdict(detected_reader_info),
            }

    def set_power(self, level: int) -> dict[str, object]:
        with self._lock:
            self._sync_runtime_state_locked()
            reader = self._require_reader()
            reader_info = self._set_power_and_refresh_locked(
                reader,
                level,
                context="manual power update",
            )
            LOGGER.info("Power changed to %s", reader_info.power)
            return self.get_status()

    def set_mode(self, mode: str) -> dict[str, object]:
        normalized_mode = self._normalize_mode(mode)
        with self._lock:
            self._processor.set_mode(normalized_mode)
            self._state.current_mode = normalized_mode
            default_power = MODE_DEFAULT_POWER[normalized_mode]
            self._state.current_power = default_power
            self._state.last_error = None
            if self._state.connected:
                self._apply_runtime_configuration_locked()
            LOGGER.info(
                "Mode changed to %s with default power %s.",
                normalized_mode,
                default_power,
            )
            return self.get_status()

    def set_transport_mode(self, mode: str) -> dict[str, object]:
        normalized_mode = self._normalize_transport_mode(mode)
        with self._lock:
            self._state.transport_mode = normalized_mode
            self._state.last_error = None
            if self._state.connected:
                self._apply_runtime_configuration_locked()
            LOGGER.info("Transport mode changed to %s.", normalized_mode)
            return self.get_status()

    def set_buzzer(self, enabled: bool) -> dict[str, object]:
        with self._lock:
            if self._state.connected and self._state.transport_mode == "answer":
                self._state.last_error = (
                    "Buzzer control is only supported in Scan transport. "
                    "Switch transport from Answer to Scan and apply again."
                )
                raise RuntimeError(self._state.last_error)
            self._state.buzzer_enabled = enabled
            self._state.last_error = None
            if self._state.connected:
                self._apply_runtime_configuration_locked()
            LOGGER.info("Buzzer changed to %s.", "enabled" if enabled else "disabled")
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

    def _normalize_transport_mode(self, mode: str) -> str:
        normalized = mode.strip().lower()
        if normalized not in {"scan", "answer"}:
            raise ValueError("Transport mode must be one of: scan, answer.")
        return normalized

    def _normalize_port_name(self, port: str) -> str:
        normalized = port.strip()
        if not normalized:
            return self._state.port
        if normalized.upper().startswith("COM"):
            return normalized.upper()
        return normalized

    def _connect_locked(
        self,
        *,
        port: str,
        baudrate: int,
        debug_raw: bool,
    ) -> dict[str, object]:
        config = ReaderConfig(
            port=port,
            baudrate=baudrate,
            debug_raw=debug_raw,
        )
        reader = SerialRFIDReader(
            config=config,
            on_tags=self._handle_detected_tags,
            on_connection_change=self._handle_connection_change,
        )
        try:
            reader.start()
            self._reader = reader
            self._supports_work_mode_readback = True
            self._processor.set_mode(self._state.current_mode)
            self._state.running = False
            self._state.connected = reader.connected
            self._needs_runtime_sync = True

            try:
                self._reader_info = reader.get_reader_info()
                self._state.current_power = self._reader_info.power
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Unable to read reader info during startup: %s", exc)

            work_mode = self._get_work_mode_if_available_locked(
                reader,
                context="startup",
            )
            if work_mode is not None:
                self._state.transport_mode = self._transport_mode_from_work_mode(work_mode)
                self._state.current_mode = self._mode_from_work_mode(work_mode)
                self._state.buzzer_enabled = work_mode.buzzer_enabled

            self._sync_runtime_state_locked()
            return self.get_status()
        except Exception:
            reader.stop()
            self._reader = None
            self._reader_info = None
            self._state.connected = False
            self._state.running = False
            self._needs_runtime_sync = False
            self._supports_work_mode_readback = True
            raise

    def _get_work_mode_if_available_locked(
        self,
        reader: SerialRFIDReader,
        *,
        context: str,
    ) -> WorkModeConfig | None:
        if not self._supports_work_mode_readback:
            return None

        try:
            return reader.get_work_mode()
        except Exception as exc:  # noqa: BLE001
            self._supports_work_mode_readback = False
            LOGGER.warning(
                "Unable to read work mode during %s: %s. Continuing without work-mode readback verification.",
                context,
                exc,
            )
            return None

    def _auto_detect_port_locked(
        self,
        *,
        preferred_port: str | None,
        baudrate: int,
        debug_raw: bool,
    ) -> tuple[SerialPortInfo, ReaderInfo] | None:
        try:
            return detect_reader_port(
                preferred_port=preferred_port,
                baudrate=baudrate,
                debug_raw=debug_raw,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("RFID auto-detect failed: %s", exc)
            return None

    def _work_mode_for(self, mode: str, transport_mode: str, buzzer_enabled: bool) -> WorkModeConfig:
        if transport_mode == "answer":
            return WorkModeConfig.answer_mode(
                mem_inven=0x05 if mode == "registration" else 0x04,
                buzzer_enabled=buzzer_enabled,
            )
        if mode == "registration":
            return WorkModeConfig.registration_scan(buzzer_enabled=buzzer_enabled)
        return WorkModeConfig.normal_scan(buzzer_enabled=buzzer_enabled)

    def _transport_mode_from_work_mode(self, mode: WorkModeConfig) -> str:
        if mode.read_mode == 0x00:
            return "answer"
        return "scan"

    def _mode_from_work_mode(self, mode: WorkModeConfig) -> str:
        if mode.mem_inven == 0x05:
            return "registration"
        return "normal"

    def _apply_runtime_configuration_locked(self) -> None:
        reader = self._require_reader()
        self._stop_answer_polling_locked()
        reader.set_work_mode(
            self._work_mode_for(
                self._state.current_mode,
                self._state.transport_mode,
                self._state.buzzer_enabled,
            )
        )
        work_mode = self._get_work_mode_if_available_locked(
            reader,
            context="runtime synchronization",
        )
        requested_power = self._state.current_power
        reader_info = self._set_power_and_refresh_locked(
            reader,
            requested_power,
            context="runtime synchronization",
        )
        if work_mode is not None:
            verified_transport = self._transport_mode_from_work_mode(work_mode)
            verified_mode = self._mode_from_work_mode(work_mode)
            verified_buzzer = work_mode.buzzer_enabled
            if verified_transport != self._state.transport_mode:
                raise RuntimeError(
                    f"Reader transport verification failed: expected {self._state.transport_mode}, got {verified_transport}."
                )
            if verified_mode != self._state.current_mode:
                raise RuntimeError(
                    f"Reader mode verification failed: expected {self._state.current_mode}, got {verified_mode}."
                )
            if verified_buzzer != self._state.buzzer_enabled:
                raise RuntimeError(
                    f"Reader buzzer verification failed: expected {self._state.buzzer_enabled}, got {verified_buzzer}."
                )
            self._state.transport_mode = verified_transport
            self._state.current_mode = verified_mode
            self._state.buzzer_enabled = verified_buzzer

        self._processor.set_mode(self._state.current_mode)
        if self._state.transport_mode == "answer" and self._state.running:
            self._ensure_answer_polling_locked()
        self._needs_runtime_sync = False
        LOGGER.info(
            "Reader runtime synchronized: mode=%s transport=%s buzzer=%s power=%s%s",
            self._state.current_mode,
            self._state.transport_mode,
            "enabled" if self._state.buzzer_enabled else "disabled",
            self._state.current_power,
            " (work mode readback unavailable)" if work_mode is None else "",
        )

    def _set_power_and_refresh_locked(
        self,
        reader: SerialRFIDReader,
        requested_power: int,
        *,
        context: str,
    ) -> ReaderInfo:
        reader.set_power(requested_power)
        reader_info = reader.get_reader_info()
        self._reader_info = reader_info
        self._state.current_power = reader_info.power

        if reader_info.power != requested_power:
            self._state.last_error = (
                f"Reader kept power {reader_info.power} after requesting {requested_power} during {context}."
            )
            LOGGER.warning(self._state.last_error)
        else:
            self._state.last_error = None

        return reader_info

    def _sync_runtime_state_locked(self) -> None:
        if not self._needs_runtime_sync or not self._state.connected:
            return
        self._apply_runtime_configuration_locked()

    def _ensure_answer_polling_locked(self) -> None:
        if not self._state.running or self._state.transport_mode != "answer":
            return
        if self._answer_poll_thread is not None and self._answer_poll_thread.is_alive():
            return

        stop_event = threading.Event()
        self._answer_poll_generation += 1
        generation = self._answer_poll_generation
        self._answer_poll_stop_event = stop_event
        self._answer_poll_thread = threading.Thread(
            target=self._answer_poll_loop,
            args=(generation, stop_event),
            name="rfid-answer-poller",
            daemon=True,
        )
        self._answer_poll_thread.start()

    def _stop_answer_polling_locked(self) -> None:
        self._answer_poll_generation += 1
        if self._answer_poll_stop_event is not None:
            self._answer_poll_stop_event.set()
        self._answer_poll_stop_event = None
        self._answer_poll_thread = None

    def _answer_poll_loop(self, generation: int, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            with self._lock:
                if (
                    generation != self._answer_poll_generation
                    or not self._state.running
                    or self._state.transport_mode != "answer"
                ):
                    return
                reader = self._reader
                current_mode = self._state.current_mode
                connected = self._state.connected

            if reader is None or not connected or not reader.connected:
                stop_event.wait(ANSWER_MODE_RETRY_SECONDS)
                continue

            try:
                packet = reader.inventory_single() if current_mode == "registration" else reader.inventory()
                tags = extract_epcs_from_packet(packet)
                if tags:
                    self._processor.process_tags(tags, raw_hex=packet.raw_hex)
                with self._lock:
                    if generation == self._answer_poll_generation:
                        self._state.last_error = None
            except TimeoutError:
                stop_event.wait(ANSWER_MODE_RETRY_SECONDS)
                continue
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Answer mode inventory failed: %s", exc)
                with self._lock:
                    if generation == self._answer_poll_generation:
                        self._state.last_error = str(exc)
                stop_event.wait(ANSWER_MODE_RETRY_SECONDS)
                continue

            stop_event.wait(ANSWER_MODE_IDLE_SECONDS)

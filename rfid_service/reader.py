from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from .parser import (
    MAX_PACKET_TOTAL_BYTES,
    MIN_PACKET_TOTAL_BYTES,
    ReaderPacket,
    build_command_frame,
    decode_packet,
    extract_epcs_from_packet,
    verify_packet_crc,
)

try:
    import serial  # type: ignore
    from serial import SerialException  # type: ignore
    from serial.tools import list_ports  # type: ignore
except ImportError:  # pragma: no cover
    serial = None
    SerialException = Exception
    list_ports = None


LOGGER = logging.getLogger("rfid_service.reader")
DEFAULT_ADDRESS = 0x00


@dataclass(slots=True)
class ReaderConfig:
    port: str = "COM3"
    baudrate: int = 57600
    address: int = DEFAULT_ADDRESS
    read_timeout: float = 0.1
    write_timeout: float = 0.5
    reconnect_delay: float = 2.0
    command_timeout: float = 1.5
    max_buffer_bytes: int = 8192
    debug_raw: bool = False


@dataclass(slots=True)
class ReaderInfo:
    version: int
    reader_type: int
    protocol_mask: int
    max_frequency: int
    min_frequency: int
    power: int
    scan_time: int


@dataclass(slots=True)
class SerialPortInfo:
    device: str
    description: str | None = None
    manufacturer: str | None = None
    hwid: str | None = None
    vid: int | None = None
    pid: int | None = None


@dataclass(slots=True)
class WorkModeConfig:
    read_mode: int
    mode_state: int
    mem_inven: int
    first_adr: int = 0x00
    word_num: int = 0x01
    tag_time: int = 0x00

    def to_payload(self) -> bytes:
        return bytes(
            [
                self.read_mode & 0xFF,
                self.mode_state & 0xFF,
                self.mem_inven & 0xFF,
                self.first_adr & 0xFF,
                self.word_num & 0xFF,
                self.tag_time & 0xFF,
            ]
        )

    @property
    def buzzer_enabled(self) -> bool:
        # Mode_state bit2: 0 = buzzer on, 1 = buzzer off.
        return (self.mode_state & 0x04) == 0

    @classmethod
    def answer_mode(cls, *, mem_inven: int = 0x04, buzzer_enabled: bool = False) -> "WorkModeConfig":
        # Answer Mode keeps the host in control of inventory polling.
        return cls(
            read_mode=0x00,
            mode_state=0x02 if buzzer_enabled else 0x06,
            mem_inven=mem_inven,
            word_num=0x06,
            tag_time=0x00,
        )

    @classmethod
    def normal_scan(cls, *, buzzer_enabled: bool = False) -> "WorkModeConfig":
        # Scan Mode + RS232 output + inventory multiple, with buzzer controlled by mode_state bit2.
        return cls(
            read_mode=0x01,
            mode_state=0x02 if buzzer_enabled else 0x06,
            mem_inven=0x04,
            word_num=0x06,
            tag_time=0x00,
        )

    @classmethod
    def registration_scan(cls, *, buzzer_enabled: bool = False) -> "WorkModeConfig":
        # Scan Mode + RS232 output + inventory single, with buzzer controlled by mode_state bit2.
        return cls(
            read_mode=0x01,
            mode_state=0x02 if buzzer_enabled else 0x06,
            mem_inven=0x05,
            word_num=0x06,
            tag_time=0x00,
        )

    @classmethod
    def from_response(cls, payload: bytes) -> "WorkModeConfig":
        if len(payload) < 10:
            raise ValueError("Get Work Mode response payload is shorter than expected.")
        return cls(
            read_mode=payload[4],
            mode_state=payload[5],
            mem_inven=payload[6],
            first_adr=payload[7],
            word_num=payload[8],
            tag_time=payload[9],
        )


def _port_sort_key(device: str) -> tuple[int, int | str]:
    normalized = device.strip().upper()
    if normalized.startswith("COM") and normalized[3:].isdigit():
        return (0, int(normalized[3:]))
    return (1, normalized)


def list_serial_ports() -> list[SerialPortInfo]:
    if list_ports is None:  # pragma: no cover
        raise RuntimeError("pyserial is required. Install rfid_service/requirements.txt first.")

    ports = [
        SerialPortInfo(
            device=port.device,
            description=getattr(port, "description", None),
            manufacturer=getattr(port, "manufacturer", None),
            hwid=getattr(port, "hwid", None),
            vid=getattr(port, "vid", None),
            pid=getattr(port, "pid", None),
        )
        for port in list_ports.comports()
    ]
    ports.sort(key=lambda port: _port_sort_key(port.device))
    return ports


def probe_reader_on_port(
    port: str,
    *,
    baudrate: int = 57600,
    debug_raw: bool = False,
) -> ReaderInfo | None:
    reader = SerialRFIDReader(
        config=ReaderConfig(
            port=port,
            baudrate=baudrate,
            read_timeout=0.1,
            write_timeout=0.3,
            reconnect_delay=0.25,
            command_timeout=0.8,
            debug_raw=debug_raw,
        ),
    )
    try:
        reader.start()
        return reader.get_reader_info()
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("RFID probe failed on %s @ %s: %s", port, baudrate, exc)
        return None
    finally:
        reader.stop()


def detect_reader_port(
    *,
    baudrate: int = 57600,
    preferred_port: str | None = None,
    debug_raw: bool = False,
) -> tuple[SerialPortInfo, ReaderInfo] | None:
    available_ports = list_serial_ports()
    if not available_ports:
        return None

    preferred = preferred_port.strip().upper() if preferred_port else None
    ordered_ports = available_ports
    if preferred:
        ordered_ports = sorted(
            available_ports,
            key=lambda port: (
                0 if port.device.strip().upper() == preferred else 1,
                *_port_sort_key(port.device),
            ),
        )

    for port_info in ordered_ports:
        reader_info = probe_reader_on_port(
            port_info.device,
            baudrate=baudrate,
            debug_raw=debug_raw,
        )
        if reader_info is not None:
            return port_info, reader_info

    return None


class SerialRFIDReader:
    def __init__(
        self,
        config: ReaderConfig,
        on_tags: Callable[[ReaderPacket, list[str]], None] | None = None,
        on_connection_change: Callable[[bool], None] | None = None,
    ) -> None:
        if serial is None:  # pragma: no cover
            raise RuntimeError("pyserial is required. Install rfid_service/requirements.txt first.")

        self._config = config
        self._on_tags = on_tags
        self._on_connection_change = on_connection_change
        self._serial: serial.Serial | None = None
        self._buffer = bytearray()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._connected = False
        self._command_lock = threading.Lock()
        self._pending_response_code: int | None = None
        self._pending_response_queue: queue.Queue[ReaderPacket] | None = None
        self._recent_packet_cache: deque[tuple[bytes, float]] = deque(maxlen=16)

    @property
    def config(self) -> ReaderConfig:
        return self._config

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._open_serial()
        self._thread = threading.Thread(target=self._read_loop, name="rfid-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._close_serial()
        self._buffer.clear()

    def set_power(self, level: int) -> None:
        if level < 0 or level > 30:
            raise ValueError("Reader power must be between 0 and 30.")
        response = self.send_command(0x2F, bytes([level]))
        if response.status != 0x00:
            raise RuntimeError(f"Reader rejected power change with status 0x{response.status:02X}.")

    def get_reader_info(self) -> ReaderInfo:
        response = self.send_command(0x21)
        if response.status != 0x00 or len(response.data) < 7:
            raise RuntimeError("Reader information response was incomplete.")
        return ReaderInfo(
            version=response.data[0],
            reader_type=response.data[1],
            protocol_mask=response.data[2],
            max_frequency=response.data[3],
            min_frequency=response.data[4],
            power=response.data[5],
            scan_time=response.data[6],
        )

    def inventory(self) -> ReaderPacket:
        return self.send_command(0x01)

    def inventory_single(self) -> ReaderPacket:
        return self.send_command(0x0F)

    def set_work_mode(self, mode: WorkModeConfig) -> None:
        response = self.send_command(0x35, mode.to_payload())
        if response.status != 0x00:
            raise RuntimeError(f"Reader rejected work mode change with status 0x{response.status:02X}.")

    def get_work_mode(self) -> WorkModeConfig:
        response = self.send_command(0x36)
        if response.status != 0x00:
            raise RuntimeError(f"Reader rejected Get Work Mode with status 0x{response.status:02X}.")
        return WorkModeConfig.from_response(response.data)

    def send_command(self, command: int, payload: bytes = b"", timeout: float | None = None) -> ReaderPacket:
        if not self.connected:
            raise RuntimeError("Reader is not connected.")

        command_frame = build_command_frame(self._config.address, command, payload)
        response_queue: queue.Queue[ReaderPacket] = queue.Queue(maxsize=1)
        effective_timeout = timeout or self._config.command_timeout

        # All host-issued commands flow through a single lock so polling,
        # configuration writes, and manual commands cannot overlap on the wire.
        with self._command_lock:
            self._pending_response_code = command
            self._pending_response_queue = response_queue
            try:
                assert self._serial is not None
                if self._config.debug_raw:
                    LOGGER.debug("TX %s", command_frame.hex().upper())
                self._serial.write(command_frame)
                self._serial.flush()
                return response_queue.get(timeout=effective_timeout)
            except queue.Empty as exc:
                raise TimeoutError(f"Timed out waiting for reader response to command 0x{command:02X}.") from exc
            finally:
                self._pending_response_code = None
                self._pending_response_queue = None

    def _set_connected(self, value: bool) -> None:
        if self._connected == value:
            return
        self._connected = value
        if self._on_connection_change:
            self._on_connection_change(value)

    def _open_serial(self) -> None:
        assert serial is not None
        try:
            self._serial = serial.Serial(
                port=self._config.port,
                baudrate=self._config.baudrate,
                timeout=self._config.read_timeout,
                write_timeout=self._config.write_timeout,
                inter_byte_timeout=self._config.read_timeout,
            )
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
            self._set_connected(True)
            LOGGER.info("Connected to RFID reader on %s @ %s baud.", self._config.port, self._config.baudrate)
        except SerialException as exc:
            self._close_serial()
            raise RuntimeError(
                f"Unable to open RFID reader on {self._config.port} @ {self._config.baudrate}: {exc}"
            ) from exc

    def _close_serial(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except SerialException:
                pass
        self._serial = None
        self._set_connected(False)

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._serial is None or not self._serial.is_open:
                try:
                    self._open_serial()
                except RuntimeError as exc:
                    LOGGER.warning("%s", exc)
                    time.sleep(self._config.reconnect_delay)
                    continue

            try:
                assert self._serial is not None
                read_size = max(1, self._serial.in_waiting or 1)
                chunk = self._serial.read(read_size)
                if not chunk:
                    time.sleep(0.01)
                    continue
                if self._config.debug_raw:
                    LOGGER.debug("RX %s", chunk.hex().upper())
                self._append_to_buffer(chunk)
                for packet in self._extract_packets():
                    self._dispatch_packet(packet)
            except SerialException as exc:
                LOGGER.warning("Reader serial error: %s", exc)
                self._close_serial()
                time.sleep(self._config.reconnect_delay)

    def _append_to_buffer(self, chunk: bytes) -> None:
        self._buffer.extend(chunk)
        if len(self._buffer) <= self._config.max_buffer_bytes:
            return
        LOGGER.warning(
            "RFID buffer exceeded %s bytes. Resetting buffer to recover framing.",
            self._config.max_buffer_bytes,
        )
        self._buffer.clear()

    def _extract_packets(self) -> list[ReaderPacket]:
        packets: list[ReaderPacket] = []
        while len(self._buffer) >= MIN_PACKET_TOTAL_BYTES:
            total_length = self._buffer[0] + 1
            if total_length < MIN_PACKET_TOTAL_BYTES or total_length > MAX_PACKET_TOTAL_BYTES:
                del self._buffer[0]
                continue
            if len(self._buffer) < total_length:
                break

            candidate = bytes(self._buffer[:total_length])
            if not verify_packet_crc(candidate):
                del self._buffer[0]
                continue

            try:
                packets.append(decode_packet(candidate))
            except ValueError:
                del self._buffer[0]
                continue

            del self._buffer[:total_length]
        return packets

    def _dispatch_packet(self, packet: ReaderPacket) -> None:
        now = time.monotonic()
        for cached_raw, cached_at in list(self._recent_packet_cache):
            if packet.raw == cached_raw and (now - cached_at) <= 0.075:
                return
        self._recent_packet_cache.append((packet.raw, now))

        if self._pending_response_code is not None and packet.response_code == self._pending_response_code:
            response_queue = self._pending_response_queue
            if response_queue is not None:
                response_queue.put(packet)
                return

        tags = extract_epcs_from_packet(packet)
        if tags and self._on_tags:
            self._on_tags(packet, tags)

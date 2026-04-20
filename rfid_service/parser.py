from __future__ import annotations

from dataclasses import dataclass


MIN_PACKET_TOTAL_BYTES = 5
MAX_PACKET_TOTAL_BYTES = 256
SPONTANEOUS_TAG_RESPONSE = 0xEE
INVENTORY_RESPONSE = 0x01
INVENTORY_SINGLE_RESPONSE = 0x0F
SUCCESS_STATUSES = {0x00, 0x01, 0x02, 0x03, 0x04}
REQUIRED_EPC_HEX_LENGTH = 8
REQUIRED_EPC_BYTES = REQUIRED_EPC_HEX_LENGTH // 2


@dataclass(slots=True)
class ReaderPacket:
    length: int
    address: int
    response_code: int
    status: int
    data: bytes
    raw: bytes

    @property
    def raw_hex(self) -> str:
        return self.raw.hex().upper()


def calculate_crc16(frame_without_crc: bytes) -> int:
    crc = 0xFFFF
    for byte in frame_without_crc:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
    return crc & 0xFFFF


def verify_packet_crc(raw_packet: bytes) -> bool:
    if len(raw_packet) < MIN_PACKET_TOTAL_BYTES:
        return False
    expected_crc = int.from_bytes(raw_packet[-2:], byteorder="little")
    calculated_crc = calculate_crc16(raw_packet[:-2])
    return expected_crc == calculated_crc


def build_command_frame(address: int, command: int, payload: bytes = b"") -> bytes:
    frame_without_crc = bytes([len(payload) + 4, address & 0xFF, command & 0xFF, *payload])
    crc = calculate_crc16(frame_without_crc)
    return frame_without_crc + crc.to_bytes(2, byteorder="little")


def decode_packet(raw_packet: bytes) -> ReaderPacket:
    if not verify_packet_crc(raw_packet):
        raise ValueError("Packet CRC mismatch.")
    length = raw_packet[0]
    payload = raw_packet[1:-2]
    if len(payload) < 3:
        raise ValueError("Packet payload is shorter than required header fields.")
    return ReaderPacket(
        length=length,
        address=payload[0],
        response_code=payload[1],
        status=payload[2],
        data=bytes(payload[3:]),
        raw=bytes(raw_packet),
    )


def is_valid_epc(epc: str) -> bool:
    return bool(epc) and epc.startswith("E2") and len(epc) == REQUIRED_EPC_HEX_LENGTH


def _normalize_epc(candidate: bytes) -> str | None:
    if len(candidate) != REQUIRED_EPC_BYTES:
        return None
    epc = candidate.hex().upper()
    if not is_valid_epc(epc):
        return None
    return epc


def _parse_declared_epcs(payload: bytes, *, starts_with_count: bool) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    cursor = 1 if starts_with_count and payload else 0

    while cursor < len(payload):
        epc_length = payload[cursor]
        cursor += 1
        if epc_length <= 0:
            continue
        if cursor + epc_length > len(payload):
            break

        candidate = _normalize_epc(payload[cursor:cursor + epc_length])
        cursor += epc_length
        if candidate and candidate not in seen:
            seen.add(candidate)
            tags.append(candidate)

    return tags


def _scan_for_embedded_epcs(payload: bytes) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    cursor = 0

    while cursor < len(payload):
        declared_length = payload[cursor]
        if declared_length != REQUIRED_EPC_BYTES:
            cursor += 1
            continue

        start = cursor + 1
        end = start + declared_length
        if end > len(payload):
            cursor += 1
            continue

        candidate = _normalize_epc(payload[start:end])
        if candidate and candidate not in seen:
            seen.add(candidate)
            tags.append(candidate)
            cursor = end
            continue

        cursor += 1

    return tags


def extract_epcs_from_packet(packet: ReaderPacket) -> list[str]:
    if packet.status not in SUCCESS_STATUSES or not packet.data:
        return []

    if packet.response_code not in {INVENTORY_RESPONSE, INVENTORY_SINGLE_RESPONSE, SPONTANEOUS_TAG_RESPONSE}:
        return []

    parsers = [
        lambda: _parse_declared_epcs(packet.data, starts_with_count=True),
        lambda: _parse_declared_epcs(packet.data, starts_with_count=False),
        lambda: _scan_for_embedded_epcs(packet.data),
    ]
    for parser in parsers:
        tags = parser()
        if tags:
            return tags
    return []

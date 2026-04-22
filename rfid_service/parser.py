from __future__ import annotations

from dataclasses import dataclass


MIN_PACKET_TOTAL_BYTES = 5
MAX_PACKET_TOTAL_BYTES = 256
SPONTANEOUS_TAG_RESPONSE = 0xEE
INVENTORY_RESPONSE = 0x01
INVENTORY_SINGLE_RESPONSE = 0x0F
SUCCESS_STATUSES = {0x00, 0x01, 0x02, 0x03, 0x04}
MAX_EPC_WORDS = 15
MIN_EPC_BYTES = 4
MAX_EPC_BYTES = MAX_EPC_WORDS * 2
MIN_EPC_HEX_LENGTH = MIN_EPC_BYTES * 2
MAX_EPC_HEX_LENGTH = MAX_EPC_BYTES * 2


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
    if len(raw_packet) != raw_packet[0] + 1:
        raise ValueError("Packet length byte does not match the raw packet size.")
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
    if not epc or len(epc) < MIN_EPC_HEX_LENGTH or len(epc) > MAX_EPC_HEX_LENGTH:
        return False
    if len(epc) % 2 != 0:
        return False
    return all(character in "0123456789ABCDEF" for character in epc)


def _normalize_epc(candidate: bytes) -> str | None:
    if len(candidate) < MIN_EPC_BYTES or len(candidate) > MAX_EPC_BYTES:
        return None
    epc = candidate.hex().upper()
    if not is_valid_epc(epc):
        return None
    return epc


def _declared_length_to_bytes(declared_length: int, *, length_units: str) -> int | None:
    if declared_length <= 0:
        return None
    if length_units == "words":
        declared_length *= 2
    if declared_length < MIN_EPC_BYTES or declared_length > MAX_EPC_BYTES:
        return None
    return declared_length


def _parse_declared_epcs(
    payload: bytes,
    *,
    starts_with_count: bool,
    length_units: str,
) -> list[str] | None:
    tags: list[str] = []
    seen: set[str] = set()
    cursor = 0
    expected_count: int | None = None
    parsed_count = 0

    if starts_with_count:
        if not payload:
            return None
        expected_count = payload[0]
        cursor = 1
        if expected_count == 0 and cursor == len(payload):
            return []

    while cursor < len(payload):
        epc_length = _declared_length_to_bytes(payload[cursor], length_units=length_units)
        cursor += 1
        if epc_length is None:
            return None
        if cursor + epc_length > len(payload):
            return None

        candidate = _normalize_epc(payload[cursor:cursor + epc_length])
        cursor += epc_length
        if not candidate:
            return None
        parsed_count += 1
        if candidate not in seen:
            seen.add(candidate)
            tags.append(candidate)

    if cursor != len(payload):
        return None
    if expected_count is not None and expected_count != parsed_count:
        return None

    return tags

def _parse_direct_epc(payload: bytes) -> list[str] | None:
    candidate = _normalize_epc(payload)
    return [candidate] if candidate else None


def _has_valid_packet_structure(packet: ReaderPacket) -> bool:
    if packet.response_code == SPONTANEOUS_TAG_RESPONSE:
        return (
            _parse_direct_epc(packet.data) is not None
            or _parse_declared_epcs(packet.data, starts_with_count=True, length_units="words") is not None
            or _parse_declared_epcs(packet.data, starts_with_count=False, length_units="words") is not None
            or _parse_declared_epcs(packet.data, starts_with_count=True, length_units="bytes") is not None
            or _parse_declared_epcs(packet.data, starts_with_count=False, length_units="bytes") is not None
        )

    if packet.response_code in {INVENTORY_RESPONSE, INVENTORY_SINGLE_RESPONSE}:
        return (
            _parse_declared_epcs(packet.data, starts_with_count=True, length_units="words") is not None
            or _parse_declared_epcs(packet.data, starts_with_count=False, length_units="words") is not None
            or _parse_declared_epcs(packet.data, starts_with_count=True, length_units="bytes") is not None
            or _parse_declared_epcs(packet.data, starts_with_count=False, length_units="bytes") is not None
            or _parse_direct_epc(packet.data) is not None
        )

    return False


def extract_epcs_from_packet(packet: ReaderPacket) -> list[str]:
    if packet.status not in SUCCESS_STATUSES or not packet.data:
        return []

    if packet.response_code not in {INVENTORY_RESPONSE, INVENTORY_SINGLE_RESPONSE, SPONTANEOUS_TAG_RESPONSE}:
        return []

    if not _has_valid_packet_structure(packet):
        return []

    if packet.response_code == SPONTANEOUS_TAG_RESPONSE:
        parsers = [
            lambda: _parse_direct_epc(packet.data),
            lambda: _parse_declared_epcs(packet.data, starts_with_count=True, length_units="words"),
            lambda: _parse_declared_epcs(packet.data, starts_with_count=False, length_units="words"),
            lambda: _parse_declared_epcs(packet.data, starts_with_count=True, length_units="bytes"),
            lambda: _parse_declared_epcs(packet.data, starts_with_count=False, length_units="bytes"),
        ]
    else:
        parsers = [
            lambda: _parse_declared_epcs(packet.data, starts_with_count=True, length_units="words"),
            lambda: _parse_declared_epcs(packet.data, starts_with_count=False, length_units="words"),
            lambda: _parse_declared_epcs(packet.data, starts_with_count=True, length_units="bytes"),
            lambda: _parse_declared_epcs(packet.data, starts_with_count=False, length_units="bytes"),
            lambda: _parse_direct_epc(packet.data),
        ]

    for parser in parsers:
        tags = parser()
        if tags:
            return tags
    return []

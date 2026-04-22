from __future__ import annotations

import unittest

from rfid_service.parser import (
    INVENTORY_SINGLE_RESPONSE,
    SPONTANEOUS_TAG_RESPONSE,
    calculate_crc16,
    decode_packet,
    extract_epcs_from_packet,
)


def build_response_packet(response_code: int, status: int, data: bytes) -> bytes:
    frame_without_crc = bytes([len(data) + 5, 0x00, response_code, status, *data])
    crc = calculate_crc16(frame_without_crc)
    return frame_without_crc + crc.to_bytes(2, byteorder="little")


class ParserTests(unittest.TestCase):
    def test_spontaneous_packet_accepts_common_12_byte_epc(self) -> None:
        epc = bytes.fromhex("300833B2DDD9014000000000")
        packet = decode_packet(build_response_packet(SPONTANEOUS_TAG_RESPONSE, 0x00, epc))

        self.assertEqual(extract_epcs_from_packet(packet), [epc.hex().upper()])

    def test_inventory_single_accepts_word_length_declared_epc(self) -> None:
        epc = bytes.fromhex("300833B2DDD9014000000000")
        # Inventory responses declare EPC length in 16-bit words on UHFReader18.
        data = bytes([0x01, len(epc) // 2, *epc])
        packet = decode_packet(build_response_packet(INVENTORY_SINGLE_RESPONSE, 0x01, data))

        self.assertEqual(extract_epcs_from_packet(packet), [epc.hex().upper()])


if __name__ == "__main__":
    unittest.main()

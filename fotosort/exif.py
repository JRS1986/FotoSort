"""Update enhanced-image geometry without relocating opaque EXIF/MakerNote data."""
from __future__ import annotations

import struct


def patch_exif_geometry(exif: bytes, size: tuple[int, int]) -> bytes:
    """Patch existing scalar tags in IFD0/ExifIFD, retaining every other byte.

    EXIF uses classic TIFF inside the six-byte Exif header. SHORT/LONG scalar
    values fit in their directory entry; changing them does not require moving
    directories or data. In particular, do not follow/rewrite MakerNotes, GPS,
    or IFD1 (the embedded thumbnail has its own, unchanged pixel geometry).
    Missing tags stay missing. Reject unsafe layouts before saving a copy.
    """
    if len(exif) < 14 or exif[:6] != b"Exif\0\0" or exif[6:8] not in (b"II", b"MM"):
        raise ValueError("Cannot safely update EXIF geometry: invalid EXIF/TIFF header")
    order = "<" if exif[6:8] == b"II" else ">"
    if struct.unpack_from(order + "H", exif, 8)[0] != 42:
        raise ValueError("Cannot safely update EXIF geometry: unsupported TIFF format")
    data = bytearray(exif)

    def directory(offset):
        start = 6 + offset
        if offset < 8 or start + 2 > len(data):
            raise ValueError("Cannot safely update EXIF geometry: invalid directory offset")
        count = struct.unpack_from(order + "H", data, start)[0]
        end = start + 2 + count * 12 + 4
        if end > len(data):
            raise ValueError("Cannot safely update EXIF geometry: truncated directory")
        return start, end, range(start + 2, end - 4, 12)

    def patch(entries, values):
        for entry in entries:
            tag, kind, count = struct.unpack_from(order + "HHI", data, entry)
            if tag not in values:
                continue
            if count != 1 or kind not in (3, 4):
                raise ValueError(f"Cannot safely update EXIF geometry: tag 0x{tag:04x} is not a SHORT/LONG scalar")
            value = values[tag]
            if not 0 <= value < 2 ** (16 if kind == 3 else 32):
                raise ValueError(f"Cannot safely update EXIF geometry: tag 0x{tag:04x} would overflow")
            struct.pack_into(order + ("H" if kind == 3 else "I"), data, entry + 8, value)

    offset = struct.unpack_from(order + "I", data, 10)[0]
    start, end, entries = directory(offset)
    exif_offsets = []
    for entry in entries:
        tag, kind, count = struct.unpack_from(order + "HHI", data, entry)
        if tag == 0x8769:
            if kind != 4 or count != 1:
                raise ValueError("Cannot safely update EXIF geometry: invalid ExifIFD pointer")
            exif_offsets.append(struct.unpack_from(order + "I", data, entry + 8)[0])
    if len(exif_offsets) > 1:
        raise ValueError("Cannot safely update EXIF geometry: duplicate ExifIFD pointers")
    if exif_offsets and exif_offsets[0]:
        child_start, child_end, child_entries = directory(exif_offsets[0])
        if child_start < end and start < child_end:
            raise ValueError("Cannot safely update EXIF geometry: overlapping directories")
        patch(child_entries, {40962: size[0], 40963: size[1]})
    patch(entries, {274: 1, 256: size[0], 257: size[1]})
    return bytes(data)

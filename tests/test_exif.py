"""EXIF layouts with TIFF-relative MakerNote offsets, generated without camera photos."""
import io
import struct

import pytest
from PIL import ExifTags, Image

from fotosort import cli
from fotosort.enhance import enhance_file, main, save_like_original
from fotosort.exif import patch_exif_geometry


def _offset_exif(order="<", scalar_type=3, *, orientation=True):
    """MakerNote data deliberately lives outside its declared block, as some cameras store it."""
    tiff = bytearray(b"~" * 2048)
    tiff[:2] = b"II" if order == "<" else b"MM"
    struct.pack_into(order + "HI", tiff, 2, 42, 48)
    locations = {}

    def directory(offset, entries, next_ifd=0, track=False):
        struct.pack_into(order + "H", tiff, offset, len(entries))
        for index, (tag, kind, count, value) in enumerate(entries):
            start = offset + 2 + index * 12
            struct.pack_into(order + "HHI", tiff, start, tag, kind, count)
            if kind == 3 and count == 1:
                tiff[start + 8:start + 12] = struct.pack(order + "H", value) + b"\xb6\xc7"
            else:
                struct.pack_into(order + "I", tiff, start + 8, value)
            if track and tag in {256, 257, 274, 40962, 40963}:
                locations[tag] = (6 + start + 8, 2 if kind == 3 else 4)
        struct.pack_into(order + "I", tiff, offset + 2 + len(entries) * 12, next_ifd)
        if track and next_ifd:
            locations["next_ifd"] = (6 + offset + 2 + len(entries) * 12, 4)

    entries = [(256, scalar_type, 1, 80), (257, scalar_type, 1, 40),
               (271, 2, 12, 272), (0x8769, 4, 1, 160), (65000, 7, 8, 560)]
    if orientation:
        entries.insert(2, (274, 3, 1, 6))
    directory(48, entries, next_ifd=640, track=True)
    directory(160, [(36867, 2, 20, 240), (0x927C, 7, 18, 400),
                    (40962, scalar_type, 1, 80), (40963, scalar_type, 1, 40)], track=True)
    tiff[240:260] = b"2026:10:02 12:00:00\0"
    tiff[272:284] = b"Test Camera\0"
    directory(400, [(1, 3, 4, 520)])
    struct.pack_into(order + "4H", tiff, 520, 8, 101, 202, 303)
    tiff[560:568] = b"UNKNOWN!"
    with io.BytesIO() as stream:
        Image.new("RGB", (8, 4), "red").save(stream, "JPEG")
        thumbnail = stream.getvalue()
    directory(640, [(256, 3, 1, 8), (257, 3, 1, 4), (259, 3, 1, 6),
                    (274, 3, 1, 6), (513, 4, 1, 768), (514, 4, 1, len(thumbnail))])
    tiff[768:768 + len(thumbnail)] = thumbnail
    return b"Exif\0\0" + bytes(tiff), locations


def _expected(raw, locations, order, size):
    expected = bytearray(raw)
    fields = ((274, 1), (256, size[0]), (257, size[1]), (40962, size[0]), (40963, size[1]), ("next_ifd", 0))
    for tag, value in fields:
        if tag in locations:
            start, length = locations[tag]
            expected[start:start + length] = value.to_bytes(length, "little" if order == "<" else "big")
    return bytes(expected)


def _check_makernote(raw, order):
    # Follow the original TIFF-relative pointer, not a newly rebuilt directory.
    pointer = struct.unpack_from(order + "I", raw, 6 + 400 + 2 + 8)[0]
    assert pointer == 520
    assert struct.unpack_from(order + "4H", raw, 6 + pointer) == (8, 101, 202, 303)


@pytest.mark.parametrize("order", ["<", ">"])
@pytest.mark.parametrize("scalar_type", [3, 4])
def test_normalized_save_preserves_all_other_exif_bytes(tmp_path, order, scalar_type):
    raw, locations = _offset_exif(order, scalar_type)
    src, dst = tmp_path / "in.jpg", tmp_path / "out.jpg"
    with Image.new("RGB", (80, 40), "blue") as original:
        original.save(src, exif=raw, icc_profile=b"test ICC")
    before = src.read_bytes()
    with Image.open(src) as original, Image.new("RGB", (30, 60), "blue") as output:
        assert original.info["exif"] == raw
        save_like_original(output, original, dst, orientation_normalized=True)
    with Image.open(dst) as saved:
        assert saved.info["exif"] == _expected(raw, locations, order, saved.size)
        _check_makernote(saved.info["exif"], order)
        assert saved.info["icc_profile"] == b"test ICC"
        assert saved.getexif()[274] == 1
        assert saved.getexif().get_ifd(0x8769)[36867] == "2026:10:02 12:00:00"
    assert src.read_bytes() == before


@pytest.mark.parametrize("state", ["upright", "rotated", "abstained", "disabled"])
def test_enhancement_keeps_original_makernote_layout(tmp_path, rightwayup_stub, state):
    raw, locations = _offset_exif()
    src, dst = tmp_path / "in.jpg", tmp_path / "out.jpg"
    with Image.new("RGB", (80, 40), "blue") as original:
        original.save(src, exif=raw)
    before = src.read_bytes()
    model = rightwayup_stub("max", "strict")
    model.angle = 12.5 if state == "rotated" else 0
    model.abstain = state == "abstained"
    details = enhance_file(src, dst, "general", 1, orienter=None if state == "disabled" else model)
    assert details["orientation_status"] == ("corrected" if state == "rotated" else state)
    with Image.open(dst) as saved:
        expected = raw if state == "disabled" else _expected(raw, locations, "<", saved.size)
        assert saved.info["exif"] == expected
        _check_makernote(saved.info["exif"], "<")
    assert src.read_bytes() == before


@pytest.mark.parametrize("order", ["<", ">"])
def test_missing_orientation_is_not_added_or_other_tags_moved(order):
    raw, locations = _offset_exif(order, orientation=False)
    result = patch_exif_geometry(raw, (30, 60))
    assert result == _expected(raw, locations, order, (30, 60))
    decoded = Image.Exif()
    decoded.load(result)
    assert 274 not in decoded
    _check_makernote(result, order)


def test_missing_geometry_tags_and_exif_directory_are_not_created():
    exif = Image.Exif()
    exif[271] = "Test Camera"
    raw = exif.tobytes()
    assert patch_exif_geometry(raw, (30, 60)) == raw


def test_geometry_patch_is_idempotent():
    raw, _ = _offset_exif()
    updated = patch_exif_geometry(raw, (30, 60))
    assert patch_exif_geometry(updated, (30, 60)) == updated


@pytest.mark.parametrize("change", [
    "header", "byte_order", "format", "root_offset", "root_truncated", "child_offset",
    "child_truncated", "overlap", "pointer_type", "pointer_count", "duplicate_pointer",
    "orientation_type", "orientation_count", "dimension_type", "dimension_count",
])
def test_unsafe_geometry_does_not_write_an_output(tmp_path, change):
    raw, locations = _offset_exif()
    data = bytearray(raw)
    # Fixture offsets are independent of the implementation under test.
    pointer_entry = 6 + 48 + 2 + 4 * 12
    if change == "header":
        data[:6] = b"wrong!"
    elif change == "byte_order":
        data[6:8] = b"??"
    elif change == "format":
        struct.pack_into("<H", data, 8, 43)  # BigTIFF is not classic EXIF
    elif change == "root_offset":
        struct.pack_into("<I", data, 10, 0xFFFFFFFF)
    elif change == "root_truncated":
        data = data[:100]
    elif change == "child_offset":
        struct.pack_into("<I", data, pointer_entry + 8, 0xFFFFFFFF)
    elif change == "child_truncated":
        data = data[:200]
    elif change == "overlap":
        struct.pack_into("<I", data, pointer_entry + 8, 48)
    elif change == "pointer_type":
        struct.pack_into("<H", data, pointer_entry + 2, 3)
    elif change == "pointer_count":
        struct.pack_into("<I", data, pointer_entry + 4, 2)
    elif change == "duplicate_pointer":
        struct.pack_into("<HHII", data, pointer_entry + 12, 0x8769, 4, 1, 160)
    else:
        tag = 274 if change.startswith("orientation") else 40962
        entry = locations[tag][0] - 8
        if change.endswith("type"):
            struct.pack_into("<H", data, entry + 2, 7)
        else:
            struct.pack_into("<I", data, entry + 4, 2)
    raw = bytes(data)
    dst = tmp_path / "out.jpg"
    with Image.new("RGB", (80, 40)) as original, Image.new("RGB", (30, 60)) as output:
        original.info["exif"] = raw
        with pytest.raises(ValueError, match="Cannot safely update EXIF geometry"):
            save_like_original(output, original, dst, orientation_normalized=True)
        assert original.info["exif"] == raw
    assert not dst.exists()


def test_scalar_overflow_is_rejected_without_rebuilding_exif():
    raw, _ = _offset_exif(scalar_type=3)
    with pytest.raises(ValueError, match="overflow"):
        patch_exif_geometry(raw, (65536, 60))


def test_truncated_header_is_rejected():
    with pytest.raises(ValueError, match="header"):
        patch_exif_geometry(b"Exif\0\0II", (30, 60))


def test_stale_thumbnail_is_unlinked_without_moving_its_bytes():
    raw, _ = _offset_exif()
    result = patch_exif_geometry(raw, (30, 60))
    before, after = Image.Exif(), Image.Exif()
    before.load(raw)
    after.load(result)
    # Viewers would show the uncorrected thumbnail with the new upright orientation.
    assert before.get_ifd(ExifTags.IFD.IFD1) and not after.get_ifd(ExifTags.IFD.IFD1)
    assert result[6 + 640:] == raw[6 + 640:]


def test_exif_pointer_with_tiff_ifd_type_is_patched():
    raw, locations = _offset_exif()
    data = bytearray(raw)
    struct.pack_into("<H", data, 6 + 48 + 2 + 4 * 12 + 2, 13)  # 0x8769 typed as IFD
    result = patch_exif_geometry(bytes(data), (30, 60))
    start, length = locations[40962]
    assert int.from_bytes(result[start:start + length], "little") == 30


def _unsafe_photo(path):
    raw, _ = _offset_exif()
    data = bytearray(raw)
    struct.pack_into("<H", data, 6 + 48 + 2 + 4 * 12 + 2, 3)  # invalid ExifIFD pointer type
    with Image.new("RGB", (80, 40), "blue") as image:
        image.save(path, exif=bytes(data))
    return path.read_bytes()


def test_unsafe_exif_error_names_the_photo_and_workaround(tmp_path, rightwayup_stub):
    src, dst = tmp_path / "in.jpg", tmp_path / "out.jpg"
    _unsafe_photo(src)
    with pytest.raises(SystemExit, match=r"in\.jpg: Cannot safely .*--no-orientation"):
        enhance_file(src, dst, "general", 1, orienter=rightwayup_stub("max", "strict"))
    assert not dst.exists()


def test_unsafe_exif_stops_export_before_relocating_any_pick(tmp_path, rightwayup_stub):
    good, bad = tmp_path / "a.jpg", tmp_path / "b.jpg"
    with Image.new("RGB", (80, 40), "blue") as image:
        image.save(good, exif=_offset_exif()[0])
    originals = {good: good.read_bytes(), bad: _unsafe_photo(bad)}
    args = cli.parse_args([str(tmp_path), "--move", "--enhance", "--enhance-style", "general"])
    with pytest.raises(SystemExit, match=r"b\.jpg: Cannot safely"):
        cli.move_picks([cli.Photo(good, "a"), cli.Photo(bad, "b")], tmp_path, args, None)
    assert all(path.read_bytes() == data for path, data in originals.items())
    assert not (tmp_path / "Highlights").exists()


def test_standalone_checks_every_photo_before_writing(tmp_path, rightwayup_stub):
    with Image.new("RGB", (80, 40), "blue") as image:
        image.save(tmp_path / "a.jpg", exif=_offset_exif()[0])
    _unsafe_photo(tmp_path / "b.jpg")
    with pytest.raises(SystemExit, match=r"b\.jpg: Cannot safely"):
        main([str(tmp_path), "--style", "general"])
    assert not (tmp_path / "Enhanced").exists()
    assert main([str(tmp_path), "--style", "general", "--no-orientation"]) == 0

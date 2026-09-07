"""Find JPEGs and read their capture time."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from PIL import Image

JPEG_EXT = {".jpg", ".jpeg", ".jpe"}
RAW_EXT = {".cr2", ".cr3", ".nef", ".nrw", ".arw", ".raf", ".orf", ".rw2", ".dng", ".pef", ".srw", ".raw"}
SIDECAR_EXT = RAW_EXT | {".xmp", ".heic"}
EXIF_IFD = 0x8769
TAG_DATETIME_ORIGINAL = 36867
TAG_DATETIME = 306
TAG_ISO = 34855


def is_raw(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXT


def find_images(root: Path, recursive: bool, exclude_dirs: set[str], include_raw: bool = True) -> list[Path]:
    """JPEGs, plus (if `include_raw`) RAW files that have no JPEG with the same
    stem next to them, so a RAW+JPEG pair is analysed once, via the JPEG."""
    exts = JPEG_EXT | (RAW_EXT if include_raw else set())
    found: list[Path] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs and not d.startswith(".")]
            for f in filenames:
                if Path(f).suffix.lower() in exts and not f.startswith("."):
                    found.append(Path(dirpath) / f)
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("."):
                found.append(p)
    jpeg_stems = {(p.parent, p.stem.lower()) for p in found if p.suffix.lower() in JPEG_EXT}
    return sorted(p for p in found if not (is_raw(p) and (p.parent, p.stem.lower()) in jpeg_stems))


def find_jpegs(root: Path, recursive: bool, exclude_dirs: set[str]) -> list[Path]:
    return find_images(root, recursive, exclude_dirs, include_raw=False)


def _raw_tags(path: Path) -> tuple[str | None, int | None]:
    import exifread  # handles TIFF-based RAW containers (ORF, NEF, CR2, ARW, DNG, ...)

    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)
    when = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
    iso = tags.get("EXIF ISOSpeedRatings")
    try:
        iso_v = int(str(iso).strip("[]").split(",")[0]) if iso else None
    except ValueError:
        iso_v = None
    return (str(when) if when else None), iso_v


def read_exif(path: Path) -> tuple[datetime | None, int | None]:
    """(capture time, ISO). Time from EXIF DateTimeOriginal, then DateTime, then
    file mtime; ISO from EXIF or None."""
    when, iso = None, None
    try:
        if is_raw(path):
            raw, iso = _raw_tags(path)
        else:
            with Image.open(path) as img:
                exif = img.getexif()
                ifd = exif.get_ifd(EXIF_IFD)
                raw = ifd.get(TAG_DATETIME_ORIGINAL) or exif.get(TAG_DATETIME)
                iso_raw = ifd.get(TAG_ISO)
                if isinstance(iso_raw, (tuple, list)):
                    iso_raw = iso_raw[0] if iso_raw else None
                iso = int(iso_raw) if iso_raw else None
        if raw:
            when = datetime.strptime(str(raw)[:19], "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    if when is None:
        try:
            when = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            when = None
    return when, iso


def capture_time(path: Path) -> datetime | None:
    return read_exif(path)[0]


def raw_sibling(path: Path) -> Path | None:
    """The RAW file belonging to a JPEG: same stem next to it or in a RAW/ subfolder."""
    if is_raw(path):
        return path
    for folder in (path.parent, path.parent / "RAW", path.parent / "raw"):
        if folder.is_dir():
            for f in folder.glob(path.stem + ".*"):
                if is_raw(f):
                    return f
    return None


def sidecars(path: Path) -> list[Path]:
    """Files next to `path` sharing its stem (RAW, XMP, ...)."""
    out = []
    for p in path.parent.glob(path.stem + ".*"):
        if p != path and p.suffix.lower() in SIDECAR_EXT:
            out.append(p)
    return out

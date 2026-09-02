"""Find JPEGs and read their capture time."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image

JPEG_EXT = {".jpg", ".jpeg", ".jpe"}
SIDECAR_EXT = {".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".dng", ".xmp", ".heic", ".raw"}
EXIF_IFD = 0x8769
TAG_DATETIME_ORIGINAL = 36867
TAG_DATETIME = 306


def find_jpegs(root: Path, recursive: bool, exclude_dirs: set[str]) -> list[Path]:
    out: list[Path] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs and not d.startswith(".")]
            for f in filenames:
                if Path(f).suffix.lower() in JPEG_EXT and not f.startswith("."):
                    out.append(Path(dirpath) / f)
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in JPEG_EXT and not p.name.startswith("."):
                out.append(p)
    return sorted(out)


def capture_time(path: Path) -> Optional[datetime]:
    """EXIF DateTimeOriginal, falling back to DateTime, then file mtime."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            raw = exif.get_ifd(EXIF_IFD).get(TAG_DATETIME_ORIGINAL) or exif.get(TAG_DATETIME)
        if raw:
            return datetime.strptime(str(raw)[:19], "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return None


def sidecars(path: Path) -> list[Path]:
    """Files next to `path` sharing its stem (RAW, XMP, ...)."""
    out = []
    for p in path.parent.glob(path.stem + ".*"):
        if p != path and p.suffix.lower() in SIDECAR_EXT:
            out.append(p)
    return out

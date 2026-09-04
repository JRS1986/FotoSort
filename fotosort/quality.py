"""Technical quality metrics: sharpness and exposure. Pure PIL + numpy."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MAX_SIDE = 1024


RAW_SUFFIXES = {".cr2", ".cr3", ".nef", ".nrw", ".arw", ".raf", ".orf", ".rw2", ".dng", ".pef", ".srw", ".raw"}


def _raw_preview(path: str) -> Image.Image:
    """The camera's embedded JPEG preview of a RAW file (fast, usually near full size)."""
    import io

    import rawpy

    with rawpy.imread(path) as raw:
        thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            return Image.open(io.BytesIO(thumb.data))
        return Image.fromarray(thumb.data)


def _raw_exif_bytes(path: str) -> bytes | None:
    """A minimal EXIF block (camera make/model, capture time) read from a RAW
    file, so a JPEG derived from it still sorts by date in an album."""
    try:
        import exifread

        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
        exif = Image.Exif()
        for tag, key in ((0x010F, "Image Make"), (0x0110, "Image Model")):
            if key in tags:
                exif[tag] = str(tags[key]).strip()
        when = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
        if when:
            exif[306] = str(when)
            exif.get_ifd(0x8769)[36867] = str(when)
        return exif.tobytes() if len(exif) else None
    except Exception:
        return None


def open_full(path: str) -> Image.Image:
    """Full-resolution RGB image: a JPEG as is, a RAW demosaiced with camera white
    balance and no auto-brightening (the enhancer does the tone work). A RAW's
    result carries a minimal EXIF (camera, capture time) in `info["exif"]`."""
    if Path(path).suffix.lower() in RAW_SUFFIXES:
        import rawpy

        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
        img = Image.fromarray(rgb)
        exif = _raw_exif_bytes(path)
        if exif:
            img.info["exif"] = exif
        return img
    img = Image.open(path)
    img.load()
    return img


def load_small(path: str, max_side: int = MAX_SIDE) -> Image.Image:
    """Decode at reduced resolution: JPEGs via DCT scaling (~5-10x faster than a
    full decode), RAW files via their embedded preview. Applies EXIF orientation."""
    if Path(path).suffix.lower() in RAW_SUFFIXES:
        img = _raw_preview(path)
    else:
        img = Image.open(path)
        img.draft("RGB", (max_side, max_side))
    img = img.convert("RGB")
    img = ImageOps.exif_transpose(img)
    img.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    return img


def to_gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)


def _laplacian(gray: np.ndarray) -> np.ndarray:
    # 4-neighbour Laplacian, valid region only
    return (
        gray[1:-1, 2:] + gray[1:-1, :-2] + gray[2:, 1:-1] + gray[:-2, 1:-1] - 4.0 * gray[1:-1, 1:-1]
    )


def sharpness(gray: np.ndarray, grid: int = 4) -> float:
    """Variance of the Laplacian, evaluated on a grid of tiles; returns the
    highest tile value. Taking the max (not the mean) rewards a sharp subject
    against a soft background, which is what a good wildlife shot looks like."""
    lap = _laplacian(gray)
    h, w = lap.shape
    if h < grid or w < grid:
        return float(lap.var())
    best = 0.0
    th, tw = h // grid, w // grid
    for i in range(grid):
        for j in range(grid):
            tile = lap[i * th : (i + 1) * th, j * tw : (j + 1) * tw]
            best = max(best, float(tile.var()))
    return best


SIG_SIDE = 24


def signature(img: Image.Image, side: int = SIG_SIDE) -> np.ndarray:
    """Tiny zero-mean, unit-norm grayscale thumbnail. The dot product of two
    signatures is a normalised cross-correlation: ~1.0 for the same framing,
    well below 0.9 for a different composition. Used for near-duplicate
    detection, where CLIP similarity is too semantic to tell frames apart."""
    a = np.asarray(img.convert("L").resize((side, side), Image.Resampling.BILINEAR), dtype=np.float32).ravel()
    a -= a.mean()
    return a / (np.linalg.norm(a) + 1e-6)


def exposure(gray: np.ndarray) -> dict:
    """Fraction of pixels clipped at the black and white ends, plus the mean."""
    n = gray.size
    return {
        "clip_low": float((gray <= 5).sum() / n),
        "clip_high": float((gray >= 250).sum() / n),
        "mean": float(gray.mean()),
    }

"""Technical quality metrics: sharpness and exposure. Pure PIL + numpy."""
from __future__ import annotations

import hashlib
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
        try:
            thumb = raw.extract_thumb()
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            return Image.fromarray(raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8))
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


def file_digest(path: str) -> str:
    """Only byte-identical sources may be removed before visual comparison."""
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def subject_region(img: Image.Image, box: tuple) -> Image.Image:
    """Crop an oriented image to a detector box, excluding its outside background."""
    w, h = img.size
    x1, y1, x2, y2 = box
    left, top = max(0, min(w - 1, int(x1 * w))), max(0, min(h - 1, int(y1 * h)))
    right, bottom = max(left + 1, min(w, int(x2 * w))), max(top + 1, min(h, int(y2 * h)))
    return img.crop((left, top, right, bottom))


def subject_focus(img: Image.Image, box: tuple) -> float | None:
    """Measure detail inside the subject, at the analysis image's common scale.

    A slight blur reduces the reward for sensor noise; the full region avoids
    letting one sharp background tile stand in for an out-of-focus animal.
    Tiny detections cannot support a reliable focus estimate.
    """
    from PIL import ImageFilter

    region = subject_region(img, box)
    if min(region.size) < 16:
        return None
    gray = to_gray(region.filter(ImageFilter.GaussianBlur(0.5)))
    return sharpness(gray, grid=1)


def subject_embedding_crop(img: Image.Image, box: tuple | None) -> Image.Image | None:
    """A subject view for local DINO comparisons; tiny detections stay unknown."""
    if box is None:
        return None
    crop = subject_region(img, box)
    return crop if min(crop.size) >= 16 else None


def detail_focus(path: str, box: tuple, side: int = 512) -> float | None:
    """Refine focus on a consistently scaled subject from the full source.

    Decode one source at a time (including RAW), respecting its orientation.
    Never upscale a tiny subject or compare native pixel scales across cameras.
    This is still a texture/noise-sensitive focus heuristic, not eye detection.
    """
    with open_full(path) as full:
        oriented = ImageOps.exif_transpose(full)
        try:
            region = subject_region(oriented, box)
        finally:
            oriented.close()
    with region:
        if max(region.size) < side or min(region.size) < 16:
            return None
        region.thumbnail((side, side), Image.Resampling.LANCZOS)
        return subject_focus(region, (0, 0, 1, 1))


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
    if min(gray.shape) < 3:
        return 0.0
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

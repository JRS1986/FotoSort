"""Technical quality metrics: sharpness and exposure. Pure PIL + numpy."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

MAX_SIDE = 1024


def load_small(path: str, max_side: int = MAX_SIDE) -> Image.Image:
    """Decode a JPEG at reduced resolution (DCT scaling), so it is ~5-10x faster
    than a full decode. Applies the EXIF orientation."""
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


def exposure(gray: np.ndarray) -> dict:
    """Fraction of pixels clipped at the black and white ends, plus the mean."""
    n = gray.size
    return {
        "clip_low": float((gray <= 5).sum() / n),
        "clip_high": float((gray >= 250).sum() / n),
        "mean": float(gray.mean()),
    }

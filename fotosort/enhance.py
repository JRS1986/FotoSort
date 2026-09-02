"""Style-aware image enhancement: detect what kind of photo it is, then apply a
matching recipe of levels, white balance, tone curve, vibrance, clarity and
sharpening. Works on the full-resolution image, never overwrites the original.

Standalone use:  python -m fotosort enhance PATH [PATH...] [--out DIR] [--style S]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# --------------------------------------------------------------------------- styles

STYLE_PROMPTS = {
    "wildlife": "a wildlife photo of an animal in nature",
    "landscape": "a landscape photo of scenery",
    "golden hour": "a photo taken at sunset, sunrise or golden hour with warm light",
    "portrait": "a portrait photo of a person or a group of people",
    "night": "a photo taken at night or in low light",
    "black and white": "a black and white monochrome photo",
    "food": "a photo of food or drinks",
    "urban": "a photo of a city street or architecture",
    "water": "a photo of the ocean, a beach or a lake",
    "macro": "a macro close-up photo of a small object, flower or insect",
    "general": "a snapshot photo",
}
STYLES = list(STYLE_PROMPTS)

# amounts are "full strength"; enhance() scales them by --strength and image stats
_BASE = dict(levels=0.6, wb=0.5, shadows=0.15, highlights=0.15, contrast=0.25, vibrance=0.25,
             saturation=0.0, clarity=0.25, sharpen=0.6, denoise=0.0, vignette=0.0, warm=0.0, bw=False)


def _recipe(**over):
    r = dict(_BASE)
    r.update(over)
    return r


RECIPES = {
    "wildlife": _recipe(contrast=0.3, clarity=0.35, vibrance=0.3, sharpen=0.8, vignette=0.15, shadows=0.2),
    "landscape": _recipe(levels=0.6, contrast=0.25, clarity=0.3, vibrance=0.35, saturation=0.05, sharpen=0.7),
    "golden hour": _recipe(wb=0.0, levels=0.4, contrast=0.2, vibrance=0.35, saturation=0.1, clarity=0.2, warm=0.04),
    "portrait": _recipe(contrast=0.15, clarity=0.1, vibrance=0.15, sharpen=0.4, vignette=0.2, shadows=0.2, wb=0.6),
    "night": _recipe(wb=0.2, levels=0.4, shadows=0.35, highlights=0.1, contrast=0.15, vibrance=0.15, clarity=0.15,
                     sharpen=0.3, denoise=1.0),
    "black and white": _recipe(bw=True, levels=0.8, contrast=0.4, clarity=0.45, sharpen=0.7, vignette=0.2,
                               vibrance=0.0, wb=0.0),
    "food": _recipe(contrast=0.25, vibrance=0.35, saturation=0.08, clarity=0.2, warm=0.03, sharpen=0.5),
    "urban": _recipe(levels=0.6, contrast=0.25, clarity=0.35, vibrance=0.2, sharpen=0.7),
    "water": _recipe(levels=0.7, contrast=0.25, clarity=0.3, vibrance=0.4, saturation=0.08, sharpen=0.6),
    "macro": _recipe(contrast=0.25, clarity=0.3, vibrance=0.3, sharpen=0.8, vignette=0.15),
    "general": _recipe(),
}

# --------------------------------------------------------------------------- helpers

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ _LUMA


def _ycc(rgb: np.ndarray):
    y = _luma(rgb)
    cb = (rgb[..., 2] - y) * 0.564
    cr = (rgb[..., 0] - y) * 0.713
    return y, cb, cr


def _from_ycc(y, cb, cr) -> np.ndarray:
    r = y + cr / 0.713
    b = y + cb / 0.564
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return np.stack([r, g, b], axis=-1)


def _apply_luma_ratio(rgb: np.ndarray, y_old: np.ndarray, y_new: np.ndarray) -> np.ndarray:
    ratio = (y_new / np.maximum(y_old, 1.0))[..., None]
    return rgb * ratio


def _blur_luma(y: np.ndarray, radius: float) -> np.ndarray:
    im = Image.fromarray(np.clip(y, 0, 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(im, dtype=np.float32)


def image_stats(rgb: np.ndarray) -> dict:
    y, cb, cr = _ycc(rgb)
    return {
        "mean": float(y.mean()),
        "std": float(y.std()),
        "chroma": float(np.sqrt(cb * cb + cr * cr).mean()),
        "warmth": float(rgb[..., 0].mean() - rgb[..., 2].mean()),
        "dark_frac": float((y < 40).mean()),
    }


# --------------------------------------------------------------------------- operations

def auto_levels(rgb: np.ndarray, strength: float, lo_pct=0.5, hi_pct=99.5, max_gain=1.4) -> np.ndarray:
    if strength <= 0:
        return rgb
    y = _luma(rgb)
    lo, hi = np.percentile(y, [lo_pct, hi_pct])
    if hi - lo < 1:
        return rgb
    gain = min(255.0 / (hi - lo), max_gain)
    stretched = (rgb - lo) * gain
    return rgb + strength * (stretched - rgb)


def white_balance(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Gray-world on the mid-tones, with capped gains so sunsets stay warm."""
    if strength <= 0:
        return rgb
    y = _luma(rgb)
    mask = (y > 30) & (y < 220)
    if mask.sum() < 100:
        return rgb
    means = rgb[mask].mean(axis=0)
    gains = np.clip(means.mean() / np.maximum(means, 1.0), 0.85, 1.18)
    gains = 1.0 + strength * (gains - 1.0)
    return rgb * gains.astype(np.float32)


def tone_curve(rgb: np.ndarray, shadows: float, highlights: float, contrast: float) -> np.ndarray:
    y = _luma(rgb)
    t = np.clip(y / 255.0, 0, 1)
    out = t
    if shadows:
        out = out + shadows * 0.6 * t * (1 - t) ** 2 * 4      # lift darks, peak around t=0.33
    if highlights:
        out = out - highlights * 0.5 * t ** 3 * (1 - t) * 4   # tame brights, peak around t=0.75
    if contrast:
        s = out * out * (3 - 2 * out)                            # smoothstep S-curve
        out = out + contrast * (s - out)
    return _apply_luma_ratio(rgb, y, np.clip(out, 0, 1) * 255.0)


def vibrance(rgb: np.ndarray, amount: float, protect_skin: bool = True) -> np.ndarray:
    """Boost muted colours more than already-saturated ones."""
    if amount <= 0:
        return rgb
    y, cb, cr = _ycc(rgb)
    c = np.sqrt(cb * cb + cr * cr)
    gain = 1.0 + amount * (1.0 - np.clip(c / 90.0, 0, 1)) ** 1.5
    if protect_skin:
        skin = (cr > 8) & (cr < 45) & (cb < -5) & (cb > -35)
        gain = np.where(skin, 1.0 + (gain - 1.0) * 0.4, gain)
    return _from_ycc(y, cb * gain, cr * gain)


def saturation(rgb: np.ndarray, amount: float) -> np.ndarray:
    if not amount:
        return rgb
    y, cb, cr = _ycc(rgb)
    return _from_ycc(y, cb * (1 + amount), cr * (1 + amount))


def clarity(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Local contrast: unsharp mask with a large radius on luminance."""
    if amount <= 0:
        return rgb
    y = _luma(rgb)
    radius = max(rgb.shape[:2]) / 80.0
    y_new = y + amount * (y - _blur_luma(y, radius))
    return _apply_luma_ratio(rgb, y, np.clip(y_new, 0, 255))


def sharpen(rgb: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return rgb
    y = _luma(rgb)
    radius = max(1.0, max(rgb.shape[:2]) / 3000.0)
    detail = y - _blur_luma(y, radius)
    detail = np.where(np.abs(detail) < 2.0, 0.0, detail)  # threshold: leave flat noise alone
    return _apply_luma_ratio(rgb, y, np.clip(y + amount * detail, 0, 255))


def denoise_chroma(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Blur colour noise while leaving luminance detail untouched."""
    if amount <= 0:
        return rgb
    y, cb, cr = _ycc(rgb)
    radius = amount * max(rgb.shape[:2]) / 1500.0
    off = 128.0
    cb = _blur_luma(cb + off, radius) - off
    cr = _blur_luma(cr + off, radius) - off
    return _from_ycc(y, cb, cr)


def vignette(rgb: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return rgb
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    factor = 1.0 - amount * np.clip(r2 / 2.0, 0, 1) ** 1.5
    return rgb * factor[..., None]


def warm(rgb: np.ndarray, amount: float) -> np.ndarray:
    if not amount:
        return rgb
    return rgb * np.array([1 + amount, 1.0, 1 - amount], dtype=np.float32)


def to_bw(rgb: np.ndarray) -> np.ndarray:
    # slight orange-filter look: skies darken, skin brightens
    y = rgb @ np.array([0.45, 0.45, 0.10], dtype=np.float32)
    return np.repeat(y[..., None], 3, axis=-1)


# --------------------------------------------------------------------------- pipeline

def adapt_recipe(recipe: dict, stats: dict, strength: float) -> dict:
    """Scale the recipe by the image's own state so it never overcooks."""
    r = dict(recipe)
    # already colourful -> less colour boost; already contrasty -> less contrast
    colour_scale = float(np.clip(1.0 - (stats["chroma"] - 25.0) / 40.0, 0.2, 1.0))
    contrast_scale = float(np.clip(1.0 - (stats["std"] - 50.0) / 40.0, 0.3, 1.0))
    r["vibrance"] *= colour_scale
    r["saturation"] *= colour_scale
    r["contrast"] *= contrast_scale
    r["clarity"] *= contrast_scale
    if abs(stats["warmth"]) > 35:
        r["wb"] *= 0.3  # strongly tinted scene (desert, sunset, blue hour): keep the cast
    if stats["mean"] < 70 and not r["bw"]:
        r["shadows"] += 0.1
    if stats["mean"] > 190:
        r["highlights"] += 0.1
    for k, v in r.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            r[k] = v * strength
    return r


def enhance(img: Image.Image, style: str, strength: float = 1.0) -> Image.Image:
    if style not in RECIPES:
        style = "general"
    if strength <= 0:
        return img.convert("RGB")
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    r = adapt_recipe(RECIPES[style], image_stats(rgb), strength)
    rgb = white_balance(rgb, r["wb"])
    rgb = auto_levels(rgb, r["levels"])
    rgb = tone_curve(rgb, r["shadows"], r["highlights"], r["contrast"])
    if r["bw"]:
        rgb = to_bw(rgb)
    else:
        rgb = vibrance(rgb, r["vibrance"], protect_skin=(style == "portrait"))
        rgb = saturation(rgb, r["saturation"])
    rgb = clarity(rgb, r["clarity"])
    rgb = denoise_chroma(rgb, r["denoise"])
    rgb = sharpen(rgb, r["sharpen"])
    rgb = vignette(rgb, r["vignette"])
    rgb = warm(rgb, r["warm"])
    return Image.fromarray(np.clip(rgb + 0.5, 0, 255).astype(np.uint8), "RGB")


def detect_style_from_stats(stats: dict) -> str:
    if stats["mean"] < 45:
        return "night"  # checked before mono: a dark shot must keep its colour
    if stats["chroma"] < 4.0:
        return "black and white"
    if stats["warmth"] > 30 and stats["chroma"] > 15:
        return "golden hour"
    return "general"


def detect_style(emb: np.ndarray | None, style_emb: np.ndarray | None, stats: dict) -> str:
    """CLIP zero-shot over STYLE_PROMPTS, with image statistics as sanity checks."""
    heuristic = detect_style_from_stats(stats)
    if emb is None or style_emb is None:
        return heuristic
    style = STYLES[int(np.argmax(emb @ style_emb.T))]
    if heuristic == "black and white":
        return heuristic  # CLIP sometimes misses mono; chroma does not
    if heuristic == "night" and stats["mean"] < 35:
        return "night"
    return style


def save_like_original(out: Image.Image, original: Image.Image, dst: Path, quality: int = 92) -> None:
    """Save with the original's EXIF and ICC profile. The pixels are processed in
    stored orientation, so the EXIF orientation tag stays valid."""
    kw = {}
    if original.info.get("exif"):
        kw["exif"] = original.info["exif"]
    if original.info.get("icc_profile"):
        kw["icc_profile"] = original.info["icc_profile"]
    out.save(dst, "JPEG", quality=quality, optimize=True, **kw)


def enhance_file(src: Path, dst: Path, style: str, strength: float) -> None:
    with Image.open(src) as im:
        im.load()
        out = enhance(im, style, strength)
        dst.parent.mkdir(parents=True, exist_ok=True)
        save_like_original(out, im, dst)


# --------------------------------------------------------------------------- standalone CLI

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fotosort enhance", description="Style-aware enhancement of JPEGs.")
    p.add_argument("paths", nargs="+", type=Path, help="JPEG files or folders")
    p.add_argument("--out", type=Path, help="Output folder (default: <folder>/Enhanced)")
    p.add_argument("--style", choices=STYLES, help="Force a style instead of detecting it")
    p.add_argument("--strength", type=float, default=1.0, help="0 = untouched, 1 = default, 1.5 = punchy")
    p.add_argument("--no-clip", action="store_true", help="Detect style from image statistics only (no model)")
    args = p.parse_args(argv)

    from fotosort.scan import find_jpegs

    files: list[Path] = []
    for path in args.paths:
        path = path.expanduser().resolve()
        if path.is_dir():
            files += find_jpegs(path, recursive=False, exclude_dirs={"Enhanced", "Highlights"})
        elif path.is_file():
            files.append(path)
    if not files:
        print("No JPEGs found.", file=sys.stderr)
        return 1

    embedder = style_emb = None
    if not args.style and not args.no_clip:
        from fotosort.embed import Embedder

        print("Loading CLIP for style detection ...")
        embedder = Embedder()
        style_emb = embedder.text_embeddings(list(STYLE_PROMPTS.values()), template="{}")

    from tqdm import tqdm

    for src in tqdm(files, unit="img", desc="Enhancing"):
        out_dir = args.out.expanduser().resolve() if args.out else src.parent / "Enhanced"
        with Image.open(src) as im:
            im.load()
            if args.style:
                style = args.style
            else:
                from fotosort.quality import load_small

                small = load_small(str(src), 512)
                stats = image_stats(np.asarray(small, dtype=np.float32))
                emb = embedder.embed_batch([embedder.prepare(small)])[0] if embedder else None
                style = detect_style(emb, style_emb, stats)
            out = enhance(im, style, args.strength)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_like_original(out, im, out_dir / src.name)
        tqdm.write(f"{src.name}: {style}")
    print(f"Enhanced {len(files)} images.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

import numpy as np
from PIL import Image

from fotosort.enhance import (
    RECIPES,
    STYLES,
    auto_levels,
    detect_style_from_stats,
    enhance,
    image_stats,
    save_like_original,
    tone_curve,
    vibrance,
)


def _gradient(w=256, h=128):
    x = np.linspace(40, 200, w, dtype=np.float32)
    rgb = np.stack([np.tile(x, (h, 1))] * 3, axis=-1)
    rgb[..., 0] *= 1.1  # a little warm
    return np.clip(rgb, 0, 255)


def test_every_style_has_a_recipe():
    assert set(STYLES) == set(RECIPES)


def test_auto_levels_stretches_dull_image():
    img = _gradient()
    out = auto_levels(img, strength=1.0)
    assert out.max() > img.max() + 20 and out.min() < img.min() - 20
    assert out.shape == img.shape


def test_vibrance_raises_chroma():
    img = _gradient()
    before = image_stats(img)["chroma"]
    after = image_stats(vibrance(img, 0.5))["chroma"]
    assert after > before * 1.2


def test_tone_curve_lifts_shadows():
    dark = np.full((32, 32, 3), 30, dtype=np.float32)
    assert tone_curve(dark, shadows=0.5, highlights=0.0, contrast=0.0).mean() > 40


def test_enhance_returns_same_size_and_changes_pixels():
    img = Image.fromarray(_gradient().astype(np.uint8))
    for style in STYLES:
        out = enhance(img, style, strength=1.0)
        assert out.size == img.size
        assert out.mode == "RGB"
        assert not np.array_equal(np.asarray(out), np.asarray(img)), style


def test_black_and_white_style_is_gray():
    img = Image.fromarray(_gradient().astype(np.uint8))
    a = np.asarray(enhance(img, "black and white", strength=1.0)).astype(int)
    assert np.abs(a[..., 0] - a[..., 1]).max() <= 1


def test_strength_zero_is_identity():
    img = Image.fromarray(_gradient().astype(np.uint8))
    out = enhance(img, "landscape", strength=0.0)
    assert np.abs(np.asarray(out).astype(int) - np.asarray(img).astype(int)).max() <= 1


def test_heuristic_style_detection():
    night = np.full((32, 32, 3), 15, dtype=np.float32)
    assert detect_style_from_stats(image_stats(night)) == "night"
    gray = np.full((32, 32, 3), 120, dtype=np.float32)
    assert detect_style_from_stats(image_stats(gray)) == "black and white"


def test_save_preserves_exif(tmp_path):
    src = tmp_path / "in.jpg"
    exif = Image.Exif()
    exif.get_ifd(0x8769)[36867] = "2026:09:01 10:00:00"
    Image.fromarray(_gradient().astype(np.uint8)).save(src, "JPEG", exif=exif.tobytes())
    with Image.open(src) as im:
        out = enhance(im, "landscape", 1.0)
        dst = tmp_path / "out.jpg"
        save_like_original(out, im, dst)
    with Image.open(dst) as im2:
        assert im2.getexif().get_ifd(0x8769).get(36867) == "2026:09:01 10:00:00"


def test_colour_photo_never_gets_black_and_white_style():
    from fotosort.enhance import STYLES, detect_style
    stats = {"chroma": 25.0, "mean": 120.0, "warmth": 5.0, "std": 50.0, "dark_frac": 0.1}
    style_emb = np.eye(len(STYLES), dtype=np.float32)
    emb = np.zeros(len(STYLES), dtype=np.float32)
    emb[STYLES.index("black and white")] = 1.0
    emb[STYLES.index("wildlife")] = 0.5
    assert detect_style(emb, style_emb, stats) == "wildlife"
    stats["chroma"] = 0.5
    assert detect_style(emb, style_emb, stats) == "black and white"

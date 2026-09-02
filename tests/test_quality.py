import numpy as np
from PIL import Image, ImageFilter

from fotosort.quality import sharpness, exposure, to_gray


def _checker(size=256, cell=8):
    y, x = np.mgrid[0:size, 0:size]
    img = (((x // cell) + (y // cell)) % 2 * 255).astype(np.uint8)
    return Image.fromarray(img).convert("RGB")


def test_sharp_image_scores_higher_than_blurred():
    sharp = _checker()
    blurred = sharp.filter(ImageFilter.GaussianBlur(4))
    assert sharpness(to_gray(sharp)) > 5 * sharpness(to_gray(blurred))


def test_sharpness_uses_sharpest_region_not_average():
    # A sharp subject in one corner with a flat background must still score high,
    # like a bokeh wildlife shot.
    bg = Image.new("RGB", (256, 256), (128, 128, 128))
    bg.paste(_checker(64), (0, 0))
    flat = Image.new("RGB", (256, 256), (128, 128, 128))
    assert sharpness(to_gray(bg)) > 100 * max(sharpness(to_gray(flat)), 1e-6)


def test_exposure_detects_clipping():
    dark = to_gray(Image.new("RGB", (64, 64), (0, 0, 0)))
    bright = to_gray(Image.new("RGB", (64, 64), (255, 255, 255)))
    mid = to_gray(Image.new("RGB", (64, 64), (128, 128, 128)))
    assert exposure(dark)["clip_low"] > 0.9
    assert exposure(bright)["clip_high"] > 0.9
    e = exposure(mid)
    assert e["clip_low"] < 0.01 and e["clip_high"] < 0.01


def test_signature_correlation_separates_same_from_different_framing():
    from fotosort.quality import signature
    a = _checker(256, 32)
    same = a.rotate(1, resample=Image.BILINEAR)
    other = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256), dtype=np.uint8)).convert("RGB")
    assert float(signature(a) @ signature(same)) > 0.93
    assert float(signature(a) @ signature(other)) < 0.5

"""Generate a synthetic photo folder with EXIF dates for end-to-end testing.
Usage: python tests/make_testset.py OUT_DIR"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

rng = np.random.default_rng(0)


def scene_image(seed: int, size=(1200, 800)) -> Image.Image:
    r = np.random.default_rng(seed)
    base = Image.new("RGB", size, tuple(int(v) for v in r.integers(40, 200, 3)))
    d = ImageDraw.Draw(base)
    for _ in range(25):
        x, y = r.integers(0, size[0]), r.integers(0, size[1])
        w, h = r.integers(30, 300), r.integers(30, 300)
        d.ellipse([x, y, x + w, y + h], fill=tuple(int(v) for v in r.integers(0, 255, 3)))
    for _ in range(300):
        x, y = r.integers(0, size[0]), r.integers(0, size[1])
        d.line([x, y, x + r.integers(-40, 40), y + r.integers(-40, 40)], fill=(0, 0, 0), width=2)
    return base


def save(img: Image.Image, path: Path, when: str):
    exif = Image.Exif()
    exif.get_ifd(0x8769)[36867] = when
    img.save(path, "JPEG", quality=85, exif=exif.tobytes())


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    # day 1: 3 bursts of the same scene (near-duplicates), one of them blurry; day 2: 2 different scenes
    plan = [
        ("2026:09:01", 0, [("10:00:%02d", 6, 0), ("10:30:%02d", 5, 3), ("15:00:%02d", 4, 0)]),
        ("2026:09:02", 1, [("08:00:%02d", 5, 0)]),
        ("2026:09:02", 2, [("17:00:%02d", 6, 2)]),
    ]
    for day, seed, bursts in plan:
        for tmpl, count, blur in bursts:
            base = scene_image(seed)
            for i in range(count):
                img = base.copy()
                # small jitter so burst frames differ slightly
                img = img.rotate(rng.normal(0, 0.5), resample=Image.BILINEAR)
                if blur:
                    img = img.filter(ImageFilter.GaussianBlur(blur))
                save(img, out / f"IMG_{n:04d}.jpg", f"{day} {tmpl % i}")
                n += 1
    # one very dark frame and one blown-out frame
    save(Image.new("RGB", (1200, 800), (2, 2, 2)), out / f"IMG_{n:04d}.jpg", "2026:09:02 18:00:00"); n += 1
    save(Image.new("RGB", (1200, 800), (254, 254, 254)), out / f"IMG_{n:04d}.jpg", "2026:09:02 18:00:01"); n += 1
    print(f"wrote {n} images to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

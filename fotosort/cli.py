"""fotosort: preselect the best, most varied photos from a folder into Highlights/."""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from fotosort import cache as cache_mod
from fotosort.group import cluster_scenes
from fotosort.labels import bucket_name, load_labels
from fotosort.quality import exposure, load_small, sharpness, signature, to_gray
from fotosort.scan import capture_time, find_jpegs, sidecars
from fotosort.select import Candidate, select_bucket


@dataclass
class Photo:
    path: Path
    key: str
    time: Optional[datetime] = None
    emb: Optional[np.ndarray] = None
    sig: Optional[np.ndarray] = None
    sharpness: float = 0.0
    clip_low: float = 0.0
    clip_high: float = 0.0
    mean: float = 0.0
    aesthetic: float = 0.0
    label: str = ""
    label_prob: float = 0.0
    scene: int = -1
    score: float = 0.0
    reject: str = ""
    selected: bool = False

    @property
    def day(self) -> str:
        return self.time.strftime("%Y-%m-%d") if self.time else "unknown-date"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="fotosort",
        description="Score JPEGs for sharpness + aesthetics, group by day and subject, "
        "and move the best few of each group into a Highlights folder.",
    )
    p.add_argument("folder", type=Path, help="Folder with JPEGs")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--move", action="store_true", help="Move picks into Highlights (default: dry run)")
    g.add_argument("--copy", action="store_true", help="Copy picks into Highlights instead of moving")
    p.add_argument("--highlights", default="Highlights", help="Name of the output subfolder")
    p.add_argument("--recursive", action="store_true", help="Also scan subfolders")
    p.add_argument("--max-per-group", type=int, default=5, help="Base number of picks per subject per day")
    p.add_argument("--extra-per", type=int, default=40,
                   help="One extra pick per this many photos in a group (0 = off); total capped at 3x --max-per-group")
    p.add_argument("--gap-seconds", type=float, default=120, help="Time gap that starts a new scene/burst")
    p.add_argument("--scene-sim", type=float, default=0.80, help="Min similarity to stay in the same scene")
    p.add_argument("--dup-sim", type=float, default=0.985, help="CLIP similarity above which two picks are near-duplicates")
    p.add_argument("--dup-pixel", type=float, default=0.93, help="Thumbnail correlation above which two picks are near-duplicates")
    p.add_argument("--aesthetic-weight", type=float, default=0.6, help="Weight of aesthetics vs sharpness (0..1)")
    p.add_argument("--blur-ratio", type=float, default=0.3, help="Reject if sharpness < ratio * day median")
    p.add_argument("--blur-floor", type=float, default=20.0, help="Reject if sharpness below this absolute value")
    p.add_argument("--labels", help="Text file with one subject label per line (overrides defaults)")
    p.add_argument("--skip-labels", default="document or screenshot", help="Comma-separated labels never selected")
    p.add_argument("--with-sidecars", action="store_true", help="Also move RAW/XMP files with the same name")
    p.add_argument("--enhance", action="store_true",
                   help="Also write style-aware enhanced versions of the picks into Highlights/Enhanced")
    p.add_argument("--enhance-strength", type=float, default=1.0, help="0 = untouched, 1 = default, 1.5 = punchy")
    p.add_argument("--enhance-style", help="Force one enhancement style for all picks instead of detecting it")
    p.add_argument("--report", default="fotosort_report.csv", help="CSV report path (relative to folder)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4, help="Decoder threads")
    p.add_argument("--no-cache", action="store_true", help="Ignore and do not write the feature cache")
    p.add_argument("--limit", type=int, help="Only process the first N files (for testing)")
    return p.parse_args(argv)


def _decode(photo: Photo, prepare):
    """Returns (photo, sharpness, exposure, tensor) or (photo, None, error, None)."""
    try:
        img = load_small(str(photo.path))
        gray = to_gray(img)
        ex = exposure(gray)
        photo.sig = signature(img)
        return photo, sharpness(gray), ex, prepare(img)
    except Exception as e:  # missing, truncated or non-JPEG file: skip, do not abort the run
        return photo, None, e, None


def compute_features(photos: list[Photo], root: Path, args):
    """Fill per-photo features (from cache or model) and return the Embedder."""
    cached = {} if args.no_cache else cache_mod.load(root)
    todo = []
    for ph in photos:
        c = cached.get(ph.key)
        if c is not None:
            ph.emb, ph.sig, ph.sharpness = c["emb"], c["sig"], c["sharpness"]
            ph.clip_low, ph.clip_high, ph.mean = c["clip_low"], c["clip_high"], c["mean"]
        else:
            todo.append(ph)
    print(f"{len(photos) - len(todo)} cached, {len(todo)} to analyse")

    from fotosort.embed import Embedder  # heavy import, only when needed

    print("Loading CLIP ViT-L/14 + aesthetic predictor ...")
    emb = Embedder()
    print(f"Running on {emb.device}")

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            batch, tensors = [], []
            futures = pool.map(lambda ph: _decode(ph, emb.prepare), todo)
            for ph, sh, ex, tensor in tqdm(futures, total=len(todo), unit="img", desc="Analysing"):
                if sh is None:
                    tqdm.write(f"Skipping {ph.path.name}: {ex}")
                    ph.reject = "unreadable"
                    continue
                ph.sharpness, ph.clip_low, ph.clip_high, ph.mean = sh, ex["clip_low"], ex["clip_high"], ex["mean"]
                batch.append(ph)
                tensors.append(tensor)
                if len(batch) >= args.batch_size:
                    for p_, e_ in zip(batch, emb.embed_batch(tensors)):
                        p_.emb = e_
                    batch, tensors = [], []
            if batch:
                for p_, e_ in zip(batch, emb.embed_batch(tensors)):
                    p_.emb = e_
        if not args.no_cache:
            for ph in todo:
                if ph.emb is None:
                    continue
                cached[ph.key] = {
                    "emb": ph.emb, "sig": ph.sig, "sharpness": ph.sharpness, "clip_low": ph.clip_low,
                    "clip_high": ph.clip_high, "mean": ph.mean,
                }
            cache_mod.save(root, cached)

    all_emb = np.stack([ph.emb for ph in photos])
    for ph, a in zip(photos, emb.aesthetic(all_emb)):
        ph.aesthetic = float(a)
    return emb


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-6 else np.zeros_like(x)


def score_and_reject(photos: list[Photo], args) -> None:
    skip = {s.strip() for s in args.skip_labels.split(",") if s.strip()}
    aest = _z(np.array([ph.aesthetic for ph in photos]))
    sharp = _z(np.log1p(np.array([ph.sharpness for ph in photos])))
    w = args.aesthetic_weight
    by_day = defaultdict(list)
    for ph in photos:
        by_day[ph.day].append(ph.sharpness)
    day_median = {d: float(np.median(v)) for d, v in by_day.items()}
    for i, ph in enumerate(photos):
        penalty = 3.0 * max(0.0, ph.clip_high - 0.02) + 2.0 * max(0.0, ph.clip_low - 0.10)
        ph.score = w * float(aest[i]) + (1 - w) * float(sharp[i]) - penalty
        if ph.sharpness < args.blur_floor or ph.sharpness < args.blur_ratio * day_median[ph.day]:
            ph.reject = "blurry"
        elif ph.clip_high > 0.4 or ph.clip_low > 0.6:
            ph.reject = "exposure"
        elif ph.label in skip:
            ph.reject = "label"


def assign_scenes_and_labels(photos: list[Photo], text_emb: np.ndarray, labels: list[str], args) -> None:
    """Cluster each day into scenes (bursts), then label every scene by its mean
    embedding so all frames of a burst land in the same subject bucket."""
    from fotosort.embed import classify

    by_day: dict[str, list[Photo]] = defaultdict(list)
    for ph in photos:
        by_day[ph.day].append(ph)
    far_future = datetime.max
    offset = 0
    for day in sorted(by_day):
        group = sorted(by_day[day], key=lambda p: (p.time or far_future, p.path.name))
        times = [p.time.timestamp() if p.time else None for p in group]
        emb = np.stack([p.emb for p in group])
        ids = cluster_scenes(times, emb, args.gap_seconds, args.scene_sim)
        for p, s in zip(group, ids):
            p.scene = offset + s
        offset += max(ids) + 1
    scenes: dict[int, list[Photo]] = defaultdict(list)
    for ph in photos:
        scenes[ph.scene].append(ph)
    scene_ids = sorted(scenes)
    means = np.stack([np.mean([p.emb for p in scenes[s]], axis=0) for s in scene_ids])
    means /= np.linalg.norm(means, axis=1, keepdims=True) + 1e-9
    names, probs = classify(means, text_emb, labels)
    for s, n, pr in zip(scene_ids, names, probs):
        for p in scenes[s]:
            p.label, p.label_prob = bucket_name(n), float(pr)


def pick_budget(n_photos: int, base: int, extra_per: int) -> int:
    """Base picks plus one per `extra_per` photos, capped at 3x base: a 450-frame
    afternoon of the same subject deserves more keepers than a 6-frame one."""
    extra = n_photos // extra_per if extra_per > 0 else 0
    return min(base + extra, 3 * base)


def select_photos(photos: list[Photo], args) -> dict[tuple[str, str], list[Photo]]:
    buckets: dict[tuple[str, str], list[Photo]] = defaultdict(list)
    for ph in photos:
        buckets[(ph.day, ph.label)].append(ph)
    for key, group in buckets.items():
        group.sort(key=lambda p: (p.scene, p.path.name))
        cands = [Candidate(str(p.path), p.score, p.scene, p.emb, p.sig) for p in group if not p.reject]
        k = pick_budget(len(cands), args.max_per_group, args.extra_per)
        chosen = set(select_bucket(cands, k, args.dup_sim, args.dup_pixel))
        for p in group:
            p.selected = str(p.path) in chosen
    return dict(sorted(buckets.items()))


def write_report(photos: list[Photo], path: Path) -> None:
    cols = ["file", "datetime", "day", "label", "label_prob", "scene", "sharpness", "clip_low",
            "clip_high", "aesthetic", "score", "reject", "selected"]
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for ph in sorted(photos, key=lambda p: (p.day, p.label, -p.score)):
            wr.writerow([
                str(ph.path), ph.time.isoformat() if ph.time else "", ph.day, ph.label,
                f"{ph.label_prob:.2f}", ph.scene, f"{ph.sharpness:.1f}", f"{ph.clip_low:.3f}",
                f"{ph.clip_high:.3f}", f"{ph.aesthetic:.2f}", f"{ph.score:.3f}", ph.reject, int(ph.selected),
            ])


def unique_dest(dest_dir: Path, src: Path, root: Path) -> Path:
    dest = dest_dir / src.name
    if dest.exists() or src.parent != root:
        rel = src.relative_to(root)
        dest = dest_dir / "__".join(rel.parts)
    n = 1
    base = dest
    while dest.exists():
        dest = base.with_name(f"{base.stem}_{n}{base.suffix}")
        n += 1
    return dest


def main(argv=None) -> int:
    args = parse_args(argv)
    root: Path = args.folder.expanduser().resolve()
    if not root.is_dir():
        print(f"Not a folder: {root}", file=sys.stderr)
        return 2
    out_dir = root / args.highlights
    files = find_jpegs(root, args.recursive, exclude_dirs={args.highlights})
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("No JPEGs found.")
        return 1
    print(f"Found {len(files)} JPEGs in {root}")

    photos = [Photo(path=f, key=cache_mod.cache_key(f, root)) for f in files]
    for ph in tqdm(photos, unit="img", desc="Reading EXIF", leave=False):
        ph.time = capture_time(ph.path)

    emb = compute_features(photos, root, args)
    unreadable = [ph for ph in photos if ph.emb is None]
    photos = [ph for ph in photos if ph.emb is not None]
    if unreadable:
        print(f"Skipped {len(unreadable)} unreadable files")
    if not photos:
        print("No readable JPEGs.")
        return 1
    labels = load_labels(args.labels)
    assign_scenes_and_labels(photos, emb.text_embeddings(labels), labels, args)
    score_and_reject(photos, args)
    buckets = select_photos(photos, args)

    report = root / args.report
    write_report(photos, report)

    picks = [ph for ph in photos if ph.selected]
    rejected = sum(1 for ph in photos if ph.reject)
    print()
    print(f"{'day':<12}{'subject':<24}{'photos':>7}{'picks':>6}")
    for (day, label), group in buckets.items():
        print(f"{day:<12}{label:<24}{len(group):>7}{sum(p.selected for p in group):>6}")
    print(f"\n{len(picks)} picks from {len(photos)} photos ({rejected} rejected as blurry/badly exposed/skipped).")
    print(f"Report: {report}")

    if not (args.move or args.copy):
        print(f"\nDry run. Add --move (or --copy) to put these into {out_dir}:")
        for ph in picks:
            print(f"  {ph.path.relative_to(root)}  [{ph.day} {ph.label}, score {ph.score:+.2f}]")
        if args.enhance:
            print(f"Enhanced versions would be written to {out_dir / 'Enhanced'}")
        return 0

    out_dir.mkdir(exist_ok=True)
    op = shutil.copy2 if args.copy else shutil.move
    moved = 0
    new_paths: dict[str, Path] = {}
    for ph in picks:
        targets = [ph.path] + (sidecars(ph.path) if args.with_sidecars else [])
        for src in targets:
            dest = unique_dest(out_dir, src, root)
            op(str(src), str(dest))
            moved += 1
            if src == ph.path:
                new_paths[str(ph.path)] = dest
    print(f"{'Copied' if args.copy else 'Moved'} {moved} files into {out_dir}")

    if args.enhance:
        enhance_picks(picks, new_paths, out_dir / "Enhanced", emb, args)
    return 0


def enhance_picks(picks: list[Photo], new_paths: dict[str, Path], enh_dir: Path, emb, args) -> None:
    from fotosort.enhance import STYLE_PROMPTS, detect_style, enhance_file, image_stats

    style_emb = None if args.enhance_style else emb.text_embeddings(list(STYLE_PROMPTS.values()), template="{}")
    enh_dir.mkdir(parents=True, exist_ok=True)
    styles = defaultdict(int)
    for ph in tqdm(picks, unit="img", desc="Enhancing"):
        src = new_paths[str(ph.path)]
        if args.enhance_style:
            style = args.enhance_style
        else:
            stats = image_stats(np.asarray(load_small(str(src), 512), dtype=np.float32))
            style = detect_style(ph.emb, style_emb, stats)
        styles[style] += 1
        enhance_file(src, enh_dir / src.name, style, args.enhance_strength)
    summary = ", ".join(f"{n} {s}" for s, n in sorted(styles.items(), key=lambda kv: -kv[1]))
    print(f"Enhanced {len(picks)} picks into {enh_dir} ({summary})")

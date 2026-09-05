"""fotosort: preselect the best, most varied photos from a folder into Highlights/."""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from fotosort import cache as cache_mod
from fotosort.group import cluster_scenes
from fotosort.judge import Verdict
from fotosort.labels import bucket_name, load_labels
from fotosort.quality import exposure, load_small, sharpness, signature, to_gray
from fotosort.scan import capture_time, find_images, is_raw, sidecars
from fotosort.selection import Candidate, ScenedCandidate, build_shortlist, chunk_for_tournament, select_day


@dataclass
class Photo:
    path: Path
    key: str
    source: Path | None = None  # file actually decoded (a DxO twin), None = path itself
    time: datetime | None = None
    emb: np.ndarray | None = None
    sig: np.ndarray | None = None
    dino: np.ndarray | None = None
    person: float = 0.0  # largest detected person box, fraction of frame
    subject: float = 0.0  # largest detected person/animal box, fraction of frame
    edge: bool = False  # that box touches the frame border
    judge_reason: str = ""
    shortlisted: bool = False
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
    p.add_argument("--dxo", nargs="?", const="all", choices=["all", "picks"],
                   help="Run RAW files through DxO PureRAW first: 'all' before analysis (denoised files are scored), "
                        "'picks' only the selected ones before enhancing/copying. Existing outputs in the DxO folder "
                        "are reused; missing ones are handed to PureRAW, where you press Process")
    p.add_argument("--dxo-app", default="PureRAW 6", help="Application name for the PureRAW hand-off")
    p.add_argument("--dxo-dir", type=Path, help="PureRAW output folder (default: a 'DxO' folder next to the RAWs)")
    p.add_argument("--dxo-wait", type=float, default=8.0,
                   help="Hours to wait for PureRAW outputs; 0 = do not hand off, use existing outputs only")
    p.add_argument("--layout", default="flat", choices=["flat", "species", "day", "day-species"],
                   help="Output folder structure: one folder, one per subject, one per day, or day/subject")
    p.add_argument("--no-raw", action="store_true",
                   help="Ignore RAW files (by default RAW files without a same-named JPEG are analysed via "
                        "their embedded preview)")
    p.add_argument("--max-per-group", type=int, default=5, help="Base number of picks per subject per day")
    p.add_argument("--extra-per", type=int, default=40,
                   help="One extra pick per this many photos in a group (0 = off); total capped at 3x --max-per-group")
    p.add_argument("--gap-seconds", type=float, default=120, help="Time gap that starts a new scene/burst")
    p.add_argument("--scene-sim", type=float, default=0.80, help="Min similarity to stay in the same scene")
    p.add_argument("--dup-sim", type=float, default=0.89,
                   help="DINOv2 similarity above which two picks are near-duplicates")
    p.add_argument("--dup-pixel", type=float, default=0.90,
                   help="Thumbnail correlation above which two picks are near-duplicates")
    p.add_argument("--min-score", type=float, default=-0.5, help="Never pick a photo scoring below this")
    p.add_argument("--subject-weight", type=float, default=0.4, help="Bonus weight for a large detected subject")
    p.add_argument("--judge", action="store_true",
                   help="Let a vision model choose the final picks from each group's shortlist")
    p.add_argument("--judge-provider", default="openai", choices=["openai", "anthropic"], help="API for --judge")
    p.add_argument("--judge-model", help="Model for --judge (default: gpt-5.6-sol / claude-opus-5)")
    p.add_argument("--judge-coverage", default="preselect", choices=["preselect", "full"],
                   help="preselect: score-based preselection of --preselect x the group's budget (best of every burst "
                        "first), then the judge tournament on those; full: the judge sees every frame")
    p.add_argument("--preselect", type=float, default=3.0,
                   help="Preselection size as a multiple of the group's budget (at least --preselect-min frames)")
    p.add_argument("--preselect-min", type=int, default=12, help="Minimum preselection size per group")
    p.add_argument("--judge-chunk", type=int, default=24, help="Frames per judge request in full coverage")
    p.add_argument("--taste-dir", help="Folder with photos you love; frames that resemble them get a score bonus")
    p.add_argument("--taste-weight", type=float, default=0.5, help="Max bonus from --taste-dir")
    p.add_argument("--judge-hint", default="",
                   help="One or two sentences about this shoot for the judge, e.g. what counts as a keeper")
    p.add_argument("--judge-detail", default="high", choices=["high", "low"],
                   help="Image detail sent to the OpenAI judge: high judges sharpness and faces, low is ~10x cheaper")
    p.add_argument("--key-file", help="File holding the API key (a .env or YAML line), instead of the environment")
    p.add_argument("--judge-base-url",
                   help="OpenAI-compatible server for a local judge, e.g. http://localhost:11434/v1 (Ollama); "
                        "requires --judge-model; no key or upload needed")
    p.add_argument("--xmp", choices=["picks", "all"],
                   help="Write XMP sidecars next to the originals: rating 5 + keywords for picks (and, with "
                        "'all', rating 3 for judged-but-not-picked, 1 for rejected frames); existing sidecars "
                        "are left alone unless --xmp-overwrite")
    p.add_argument("--xmp-overwrite", action="store_true", help="Replace existing .xmp sidecars")
    p.add_argument("--detector", default="yolov8m.pt", help="YOLOv8 weights for subject detection (n/s/m/l)")
    p.add_argument("--aesthetic-weight", type=float, default=0.6, help="Weight of aesthetics vs sharpness (0..1)")
    p.add_argument("--blur-ratio", type=float, default=0.15, help="Reject if sharpness < ratio * day median")
    p.add_argument("--blur-floor", type=float, default=20.0, help="Reject if sharpness below this absolute value")
    p.add_argument("--labels", help="Text file with one subject label per line (overrides defaults)")
    p.add_argument("--label-mode", default="scene", choices=["scene", "image"],
                   help="scene: one subject label per burst (steady for walking safaris); image: label every frame "
                        "on its own (use on a boat or anywhere the background never changes)")
    p.add_argument("--person-area", type=float, default=0.02,
                   help="A detected person covering at least this fraction of the frame puts the photo in the "
                        "'people' bucket (0 = disable the detector)")
    p.add_argument("--skip-labels", default="document or screenshot", help="Comma-separated labels never selected")
    p.add_argument("--with-sidecars", action="store_true", help="Also move RAW/XMP files with the same name")
    p.add_argument("--enhance", action="store_true",
                   help="Write style-aware enhanced versions of the picks. With --copy only the enhanced "
                        "version goes into the output folder (originals stay where they are); with --move "
                        "the original is moved there and the enhanced version goes into an Enhanced/ subfolder")
    p.add_argument("--enhance-strength", type=float, default=1.0, help="0 = untouched, 1 = default, 1.5 = punchy")
    p.add_argument("--enhance-style", help="Force one enhancement style for all picks instead of detecting it")
    p.add_argument("--report", default="fotosort_report.csv", help="CSV report path (relative to folder)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4, help="Decoder threads")
    p.add_argument("--no-cache", action="store_true", help="Ignore and do not write the feature cache")
    p.add_argument("--limit", type=int, help="Only process the first N files (for testing)")
    p.add_argument("--apply-report", action="store_true",
                   help="Skip analysis: take the picks (selected=1) from the existing report and copy/move/enhance "
                        "them. Edit the CSV first to override the tool's choices.")
    return p.parse_args(argv)


def _decode(photo: Photo, prepare):
    """Returns (photo, sharpness, exposure, tensor, small image) or (photo, None, error, None, None)."""
    try:
        img = load_small(str(photo.source or photo.path))
        gray = to_gray(img)
        ex = exposure(gray)
        photo.sig = signature(img)
        small = img.copy()
        small.thumbnail((640, 640))
        return photo, sharpness(gray), ex, prepare(img), small
    except Exception as e:  # missing, truncated or non-JPEG file: skip, do not abort the run
        return photo, None, e, None, None


def compute_features(photos: list[Photo], root: Path, args):
    """Fill per-photo features (from cache or model) and return the Embedder."""
    det_name = args.detector if args.person_area > 0 else "none"
    cached = {} if args.no_cache else cache_mod.load(root, det_name)
    todo = []
    for ph in photos:
        c = cached.get(ph.key)
        if c is not None:
            ph.emb, ph.sig, ph.dino = c["emb"], c["sig"], c["dino"]
            ph.person, ph.sharpness = c["person"], c["sharpness"]
            ph.subject, ph.edge = c["subject"], c["edge"]
            ph.clip_low, ph.clip_high, ph.mean = c["clip_low"], c["clip_high"], c["mean"]
        else:
            todo.append(ph)
    print(f"{len(photos) - len(todo)} cached, {len(todo)} to analyse")

    from fotosort.embed import Embedder  # heavy import, only when needed

    print("Loading CLIP ViT-L/14 + aesthetic predictor ...")
    emb = Embedder()
    print(f"Running on {emb.device}")

    detector = None
    if todo:
        if args.person_area > 0:
            from fotosort.embed import SubjectDetector

            detector = SubjectDetector(device=emb.device, weights=args.detector)

        from fotosort.embed import DinoEmbedder

        dino = DinoEmbedder(device=emb.device)

        def flush(batch, tensors, smalls):
            for p_, e_ in zip(batch, emb.embed_batch(tensors), strict=True):
                p_.emb = e_
            for p_, d_ in zip(batch, dino.embed_batch(smalls), strict=True):
                p_.dino = d_
            if detector is not None:
                for p_, d_ in zip(batch, detector.detect(smalls), strict=True):
                    p_.person, p_.subject, p_.edge = d_["person"], d_["subject"], d_["edge"]

        window = 4 * args.batch_size  # bounds how far the decoders run ahead of the GPU (memory)
        with ThreadPoolExecutor(max_workers=args.workers) as pool, \
                tqdm(total=len(todo), unit="img", desc="Analysing") as bar:
            batch, tensors, smalls = [], [], []
            for start in range(0, len(todo), window):
                chunk = todo[start:start + window]
                for ph, sh, ex, tensor, small in pool.map(lambda ph: _decode(ph, emb.prepare), chunk):
                    bar.update(1)
                    if sh is None:
                        tqdm.write(f"Skipping {ph.path.name}: {ex}")
                        ph.reject = "unreadable"
                        continue
                    ph.sharpness, ph.clip_low, ph.clip_high, ph.mean = sh, ex["clip_low"], ex["clip_high"], ex["mean"]
                    batch.append(ph)
                    tensors.append(tensor)
                    smalls.append(small)
                    if len(batch) >= args.batch_size:
                        flush(batch, tensors, smalls)
                        batch, tensors, smalls = [], [], []
            if batch:
                flush(batch, tensors, smalls)
        if not args.no_cache:
            for ph in todo:
                if ph.emb is None:
                    continue
                cached[ph.key] = {
                    "emb": ph.emb, "sig": ph.sig, "dino": ph.dino,
                    "person": ph.person, "subject": ph.subject, "edge": ph.edge,
                    "sharpness": ph.sharpness,
                    "clip_low": ph.clip_low,
                    "clip_high": ph.clip_high, "mean": ph.mean,
                }
            cache_mod.save(root, cached, det_name)

    readable = [ph for ph in photos if ph.emb is not None]
    if readable:
        all_emb = np.stack([ph.emb for ph in readable])
        for ph, a in zip(readable, emb.aesthetic(all_emb), strict=True):
            ph.aesthetic = float(a)
    emb.detector = detector if todo else None  # reused by the judge for subject crops
    return emb


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-6 else np.zeros_like(x)


def taste_bonus(photos: list[Photo], taste_dir: str, weight: float, emb) -> None:
    """Score bonus for resembling the user's own favourites: CLIP similarity to
    the closest reference photo, mapped from 0.6 (unrelated) to 0.9 (same kind
    of shot) onto 0..weight. Adds to score before selection and shortlisting."""
    refs = find_images(Path(taste_dir).expanduser().resolve(), recursive=False, exclude_dirs=set())
    if not refs:
        print(f"No JPEGs in --taste-dir {taste_dir}")
        return
    tensors = []
    for r in refs:
        try:
            tensors.append(emb.prepare(load_small(str(r))))
        except OSError as e:
            print(f"Skipping reference {r.name}: {e}")
    if not tensors:
        return
    ref_emb = emb.embed_batch(tensors)
    for ph in photos:
        sim = float(np.max(ref_emb @ ph.emb))
        ph.score += weight * float(np.clip((sim - 0.6) / 0.3, 0.0, 1.0))
    print(f"Taste bonus from {len(refs)} reference photos applied")


def subject_term(ph: Photo, weight: float) -> float:
    """Reward a clearly visible subject, punish a small one cut off by the frame.
    0 for no detection or a subject under 1 % of the frame; +1.5*weight at 30 %+.
    A small subject touching the border (walking out of the picture) costs weight."""
    if ph.subject <= 0:
        return 0.0
    size = float(np.clip(np.log10(ph.subject / 0.01), 0.0, 1.5))
    cut = 1.0 if (ph.edge and ph.subject < 0.15) else 0.0
    return weight * (size - cut)


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
        ph.score = w * float(aest[i]) + (1 - w) * float(sharp[i]) - penalty + subject_term(ph, args.subject_weight)
        # the Laplacian measure is texture-dependent (a smooth white gull reads as soft), so with a judge
        # on only the absolute floor rejects; the judge sees a 100% crop and decides sharpness itself
        ratio = 0.0 if args.judge else args.blur_ratio
        if ph.sharpness < args.blur_floor or ph.sharpness < ratio * day_median[ph.day]:
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
        for p, s in zip(group, ids, strict=True):
            p.scene = offset + s
        offset += max(ids) + 1
    if args.label_mode == "image":
        names, probs = classify(np.stack([p.emb for p in photos]), text_emb, labels)
        for p, n, pr in zip(photos, names, probs, strict=True):
            p.label, p.label_prob = bucket_name(n), float(pr)
    else:
        scenes: dict[int, list[Photo]] = defaultdict(list)
        for ph in photos:
            scenes[ph.scene].append(ph)
        scene_ids = sorted(scenes)
        means = np.stack([np.mean([p.emb for p in scenes[s]], axis=0) for s in scene_ids])
        means /= np.linalg.norm(means, axis=1, keepdims=True) + 1e-9
        names, probs = classify(means, text_emb, labels)
        for s, n, pr in zip(scene_ids, names, probs, strict=True):
            for p in scenes[s]:
                p.label, p.label_prob = bucket_name(n), float(pr)
    apply_person_override(photos, args.person_area)


def apply_person_override(photos: list[Photo], min_area: float) -> None:
    """A clearly visible person beats whatever CLIP thought the scene was."""
    if min_area <= 0:
        return
    for p in photos:
        if p.person >= min_area:
            p.label, p.label_prob = "people", max(p.label_prob, p.person)


def judge_boxes(photos: list[Photo], judge) -> dict[str, tuple | None]:
    """Subject boxes for the judge's 100% crops, detected on the fly for the shortlist only."""
    det = getattr(judge, "detector", None)
    if det is None:
        return {}
    smalls, kept = [], []
    for p in photos:
        try:
            im = load_small(str(p.source or p.path))
        except OSError:  # deleted while we were running
            continue
        im.thumbnail((640, 640))
        smalls.append(im)
        kept.append(p)
    return {str(p.path): d["box"] for p, d in zip(kept, det.detect(smalls), strict=True)}


def pick_budget(n_photos: int, base: int, extra_per: int) -> int:
    """Base picks plus one per `extra_per` photos, capped at 3x base: a 450-frame
    afternoon of the same subject deserves more keepers than a 6-frame one."""
    extra = n_photos // extra_per if extra_per > 0 else 0
    return min(base + extra, 3 * base)


def select_photos(photos: list[Photo], args, judge=None) -> dict[tuple[str, str], list[Photo]]:
    buckets: dict[tuple[str, str], list[Photo]] = defaultdict(list)
    for ph in photos:
        buckets[(ph.day, ph.label)].append(ph)
    by_path = {str(p.path): p for p in photos}
    days = sorted({d for d, _ in buckets})
    for day in days:
        day_buckets = {label: group for (d, label), group in buckets.items() if d == day}
        budgets = {label: pick_budget(sum(1 for p in g if not p.reject), args.max_per_group, args.extra_per)
                   for label, g in day_buckets.items()}
        cands = [Candidate(str(p.path), p.score, p.label, p.emb, p.sig, p.dino)
                 for g in day_buckets.values() for p in g if not p.reject]
        if judge is None:
            chosen = set(select_day(cands, budgets, args.dup_sim, args.dup_pixel, args.min_score))
        else:
            chosen = set()
            for label, k in budgets.items():
                group = [p for p in day_buckets[label] if not p.reject]
                sc = [ScenedCandidate(str(p.path), p.score, p.label, p.emb, p.sig, p.dino, p.scene) for p in group]
                if args.judge_coverage == "preselect":
                    keep = set(build_shortlist(sc, max(args.preselect_min, int(round(args.preselect * k)))))
                    sc = [c for c in sc if c.id in keep]
                rounds, twins = chunk_for_tournament(sc, args.judge_chunk)
                for dropped, kept_id in twins.items():
                    by_path[dropped].judge_reason = f"= twin of {Path(kept_id).name}"
                rounds = [[i for i in r if Path(i).exists()] for r in rounds]  # user may delete files meanwhile
                rounds = [r for r in rounds if r]
                if not rounds:
                    continue
                n_total = sum(map(len, rounds))
                finalists: list[str] = []
                for ids in rounds:
                    for i in ids:
                        by_path[i].shortlisted = True
                    # each chunk may send on its share of the budget, at least 2, so a strong burst is not capped early
                    kk = k if len(rounds) == 1 else min(k, max(2, -(-k * len(ids) // n_total) + 1))
                    boxes = judge_boxes([by_path[i] for i in ids], judge)
                    verdict = judge.judge([Path(i) for i in ids], label, day, kk, boxes)
                    if verdict.error:
                        tqdm.write(f"Judge failed for {day} {label} ({verdict.error}); using scores instead")
                        judge.failures = getattr(judge, "failures", 0) + 1
                        if judge.failures >= 3:
                            raise SystemExit("The judge failed three times in a row; check the API key, model name "
                                             "and network, or run without --judge.")
                        picks = sorted(ids, key=lambda i: by_path[i].score, reverse=True)[:kk]
                        verdict = Verdict(picks, {i: "(judge failed, by score)" for i in picks})
                    else:
                        judge.failures = 0
                    for i in verdict.picks[:kk]:
                        finalists.append(i)
                        by_path[i].judge_reason = verdict.reasons.get(i, "")
                # knock-out rounds: while too many finalists for one request, judge them in chunks again
                while len(finalists) > max(k, args.judge_chunk):
                    n_chunks = -(-len(finalists) // args.judge_chunk)
                    size = -(-len(finalists) // n_chunks)
                    nxt: list[str] = []
                    for i0 in range(0, len(finalists), size):
                        ids = finalists[i0:i0 + size]
                        kk = min(k, max(2, -(-k * len(ids) // len(finalists)) + 1))
                        verdict = judge.judge([Path(i) for i in ids], label, day, kk,
                                              judge_boxes([by_path[i] for i in ids], judge), final=True)
                        if verdict.error or not verdict.picks:
                            picks = sorted(ids, key=lambda i: by_path[i].score, reverse=True)[:kk]
                        else:
                            picks = verdict.picks[:kk]
                        for i in picks:
                            by_path[i].judge_reason = verdict.reasons.get(i) or by_path[i].judge_reason
                        nxt += picks
                    if len(nxt) >= len(finalists):
                        break
                    finalists = nxt
                if len(finalists) > k:  # final round among the remaining winners
                    verdict = judge.judge([Path(i) for i in finalists], label, day, k,
                                          judge_boxes([by_path[i] for i in finalists], judge), final=True)
                    if not verdict.error and verdict.picks:
                        for i in verdict.picks[:k]:
                            by_path[i].judge_reason = verdict.reasons.get(i) or by_path[i].judge_reason
                        finalists = verdict.picks[:k]
                    else:
                        finalists = finalists[:k]
                chosen.update(finalists)
        for p in by_path.values():
            if p.day == day:
                p.selected = str(p.path) in chosen
    return dict(sorted(buckets.items()))


def write_report(photos: list[Photo], path: Path) -> None:
    cols = ["file", "datetime", "day", "label", "label_prob", "person", "subject", "edge", "scene", "sharpness",
            "clip_low", "clip_high", "aesthetic", "score", "reject", "shortlisted", "selected", "judge_reason"]
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for ph in sorted(photos, key=lambda p: (p.day, p.label, -p.score)):
            wr.writerow([
                str(ph.path), ph.time.isoformat() if ph.time else "", ph.day, ph.label,
                f"{ph.label_prob:.2f}", f"{ph.person:.3f}", f"{ph.subject:.3f}", int(ph.edge), ph.scene,
                f"{ph.sharpness:.1f}", f"{ph.clip_low:.3f}", f"{ph.clip_high:.3f}", f"{ph.aesthetic:.2f}",
                f"{ph.score:.3f}", ph.reject, int(ph.shortlisted), int(ph.selected), ph.judge_reason,
            ])


def write_sidecars(photos: list[Photo], args) -> None:
    from fotosort.xmp import write_sidecar

    written = skipped = 0
    for ph in photos:
        if ph.selected:
            rating, label = 5, "Green"
        elif args.xmp == "all" and ph.reject:
            rating, label = 1, "Red"
        elif args.xmp == "all" and ph.shortlisted:
            rating, label = 3, ""
        elif args.xmp == "all":
            rating, label = 0, ""
        else:
            continue
        keywords = ["FotoSort", f"FotoSort|{ph.label}"] + (["FotoSort|Pick"] if ph.selected else [])
        try:
            ok = write_sidecar(ph.path, rating, keywords, ph.judge_reason if ph.selected else "", label,
                               overwrite=args.xmp_overwrite)
        except OSError as e:
            tqdm.write(f"Could not write sidecar for {ph.path.name}: {e}")
            continue
        written += ok
        skipped += not ok
    print(f"XMP sidecars: {written} written" + (f", {skipped} existing left untouched" if skipped else ""))


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
    files = find_images(root, args.recursive, exclude_dirs={args.highlights}, include_raw=not args.no_raw)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("No JPEG or RAW files found.")
        return 1
    n_raw = sum(1 for f in files if is_raw(f))
    print(f"Found {len(files)} images in {root}" + (f" ({n_raw} RAW without a JPEG twin)" if n_raw else ""))

    photos = [Photo(path=f, key=cache_mod.cache_key(f, root)) for f in files]
    if args.dxo == "all" and not args.apply_report:
        attach_dxo(photos, root, args, only=None)
    for ph in tqdm(photos, unit="img", desc="Reading EXIF", leave=False):
        ph.time = capture_time(ph.path)

    if args.apply_report:
        return apply_report(photos, root, args)

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
    if args.taste_dir:
        taste_bonus(photos, args.taste_dir, args.taste_weight, emb)
    judge = None
    if args.judge:
        from fotosort.judge import Judge

        judge = Judge(args.judge_provider, args.judge_model, args.key_file, args.judge_detail, args.judge_hint,
                      args.judge_base_url)
        judge.detector = getattr(emb, "detector", None)
        if judge.detector is None and args.person_area > 0:
            from fotosort.embed import SubjectDetector

            judge.detector = SubjectDetector(device=emb.device, weights=args.detector)
        print(f"Judging shortlists with {judge.model} ...")
    buckets = select_photos(photos, args, judge)
    if judge is not None:
        print(f"Judge tokens: {judge.usage['input']} in, {judge.usage['output']} out")

    report = root / args.report
    write_report(photos, report)
    if args.xmp:
        write_sidecars(photos, args)

    picks = [ph for ph in photos if ph.selected]
    if args.dxo == "picks" and (args.move or args.copy):
        attach_dxo(photos, root, args, only=picks)
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
            where = out_dir if args.copy else out_dir / "Enhanced"
            print(f"Enhanced versions would be written to {where}")
        return 0

    return move_picks(picks, root, args, emb)


def layout_dir(out_dir: Path, ph: Photo, layout: str) -> Path:
    """Where a pick goes: flat, per subject, per day, or day/subject."""
    safe = "".join(c if c.isalnum() or c in " -_()" else "_" for c in ph.label).strip() or "other"
    parts = {"flat": [], "species": [safe], "day": [ph.day], "day-species": [ph.day, safe]}[layout]
    return out_dir.joinpath(*parts)


def move_picks(picks: list[Photo], root: Path, args, emb) -> int:
    """Copy or move the picks (plus sidecars and DxO twins) into the output
    layout, then enhance. With --copy --enhance the enhanced JPEG is the copy;
    with --move --enhance the originals move and the enhanced versions go into
    an Enhanced/ subfolder of each destination folder."""
    out_dir = root / args.highlights
    out_dir.mkdir(exist_ok=True)
    enhanced_only = args.enhance and args.copy
    op = shutil.copy2 if args.copy else shutil.move
    moved = 0
    picks = [ph for ph in picks if ph.path.exists()]
    for ph in picks:
        dest_dir = layout_dir(out_dir, ph, args.layout)
        dest_dir.mkdir(parents=True, exist_ok=True)
        targets = sidecars(ph.path) if args.with_sidecars else []
        if ph.source and ph.source != ph.path and ph.source.exists():
            targets.append(ph.source)  # the DxO-processed file travels with the RAW
        if not enhanced_only:
            targets = [ph.path] + targets
        for src in targets:
            # a DxO twin keeps its own name (it lives in a subfolder, which unique_dest would otherwise prefix)
            dest = unique_dest(dest_dir, src, src.parent if src == ph.source else root)
            op(str(src), str(dest))
            moved += 1
            if src == ph.path:
                ph.path = dest  # keep pointing at the file where it now lives
            elif src == ph.source:
                ph.source = dest
    if moved:
        print(f"{'Copied' if args.copy else 'Moved'} {moved} files into {out_dir}")

    if args.enhance:
        def dest_of(ph: Photo) -> Path:
            base = layout_dir(out_dir, ph, args.layout)
            if not enhanced_only:
                base = base / "Enhanced"
            base.mkdir(parents=True, exist_ok=True)
            return unique_dest(base, _jpg_name(ph.path), root)

        enhance_picks(picks, emb, args, dest_of, out_dir)
    return 0


def attach_dxo(photos: list[Photo], root: Path, args, only: list[Photo] | None) -> None:
    """Find or produce PureRAW twins for the RAW files in `only` (None = all)
    and make them the decode source. Handing off to PureRAW is skipped when
    --dxo-wait is 0."""
    from fotosort.dxo import find_twin, send_to_pureraw, wait_for_twins

    group = [ph for ph in (only if only is not None else photos) if is_raw(ph.path)]
    if not group:
        return
    missing = []
    for ph in group:
        twin = find_twin(ph.path, args.dxo_dir)
        if twin is not None:
            ph.source = twin
        else:
            missing.append(ph)
    print(f"DxO PureRAW: {len(group) - len(missing)} of {len(group)} RAW files already processed")
    if missing and args.dxo_wait > 0:
        print(f"Handing {len(missing)} RAW files to {args.dxo_app}: press Process there. "
              f"Waiting up to {args.dxo_wait:g} h for the outputs ...")
        try:
            send_to_pureraw([ph.path for ph in missing], args.dxo_app)
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"Could not open {args.dxo_app}: {e}; continuing with the originals")
        else:
            done = wait_for_twins([ph.path for ph in missing], args.dxo_dir, args.dxo_wait * 3600)
            for ph in missing:
                if ph.path in done:
                    ph.source = done[ph.path]
            left = len(missing) - len(done)
            if left:
                print(f"{left} RAW files still unprocessed; using the originals for those")
    elif missing:
        print(f"{len(missing)} RAW files have no DxO output yet; using the originals for those")
    for ph in group:
        if ph.source is not None:
            ph.key = cache_mod.cache_key(ph.source, root)


def apply_report(photos: list[Photo], root: Path, args) -> int:
    """Copy/move/enhance whatever the existing report marks selected=1."""
    report = root / args.report
    if not report.exists():
        print(f"No report at {report}; run without --apply-report first.", file=sys.stderr)
        return 2
    with open(report, newline="") as f:
        rows = {r["file"]: r for r in csv.DictReader(f)}
    picks = []
    for ph in photos:
        r = rows.get(str(ph.path))
        if r and r.get("selected") == "1":
            ph.person = float(r.get("person") or 0)
            ph.label, ph.selected = r.get("label", ""), True
            picks.append(ph)
    missing = sum(1 for f, r in rows.items() if r.get("selected") == "1" and f not in {str(p.path) for p in picks})
    print(f"{len(picks)} picks from the report" + (f" ({missing} listed files no longer exist)" if missing else ""))
    if not (args.move or args.copy):
        for ph in picks:
            print(f"  {ph.path.relative_to(root)}  [{ph.label}]")
        print("Dry run. Add --move or --copy.")
        return 0
    if args.dxo:
        attach_dxo(photos, root, args, only=picks)
    emb = None
    if args.enhance and not args.enhance_style:
        from fotosort.embed import Embedder

        emb = Embedder()  # for style detection of non-people picks
        cached = cache_mod.load(root)
        for ph in picks:
            c = cached.get(ph.key)
            ph.emb = c["emb"] if c else None
    return move_picks(picks, root, args, emb)


def _jpg_name(src: Path) -> Path:
    """Enhanced output of a RAW file is a JPEG; a JPEG keeps its name."""
    return src.with_suffix(".jpg") if is_raw(src) else src


def enhance_picks(picks: list[Photo], emb, args, dest_of, out_dir: Path) -> None:
    """Enhance each pick from its source (the DxO twin if there is one) into `dest_of(pick)`."""
    from fotosort.enhance import STYLE_PROMPTS, detect_style, enhance_file, image_stats

    style_emb = None
    if not args.enhance_style and emb is not None:
        style_emb = emb.text_embeddings(list(STYLE_PROMPTS.values()), template="{}")
    styles = defaultdict(int)
    for ph in tqdm(picks, unit="img", desc="Enhancing"):
        src = ph.source or ph.path
        if args.enhance_style:
            style = args.enhance_style
        elif ph.person >= args.person_area > 0:
            style = "portrait"  # a visible person: gentle tone curve, protected skin, no heavy clarity
        else:
            stats = image_stats(np.asarray(load_small(str(src), 512), dtype=np.float32))
            style = detect_style(ph.emb if style_emb is not None else None, style_emb, stats)
        styles[style] += 1
        enhance_file(src, dest_of(ph), style, args.enhance_strength)
    summary = ", ".join(f"{n} {s}" for s, n in sorted(styles.items(), key=lambda kv: -kv[1]))
    print(f"Enhanced {len(picks)} picks into {out_dir} ({summary})")

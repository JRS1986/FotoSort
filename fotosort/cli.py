"""fotosort: preselect the best, most varied photos from a folder into Highlights/."""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from fotosort import cache as cache_mod
from fotosort.award import AwardAssessment, award_shortlist, award_work_bound, checked_award_picks
from fotosort.burst import BURST_PROMPTS, burst_scores, local_burst_views, valid_boxes
from fotosort.composition import EDITORIAL_PROMPTS, GEOMETRY_FIELDS, composition_metrics, editorial_scores
from fotosort.editing import STYLE_TO_PRESET, benefit_band, edit_benefit
from fotosort.group import cluster_bursts, cluster_moments, cluster_scenes
from fotosort.judge import Verdict
from fotosort.labels import bucket_name, is_people_label, load_labels
from fotosort.quality import (
    detail_focus,
    exposure,
    file_digest,
    load_small,
    sharpness,
    signature,
    subject_embedding_crop,
    subject_focus,
    to_gray,
)
from fotosort.review_data import (
    REVIEW_FIELDS,
    ReviewError,
    ReviewStore,
    apply_manual_decisions,
    atomic_write,
    fingerprint,
    photo_id,
)
from fotosort.scan import find_images, is_raw, raw_sibling, read_exif, sidecars
from fotosort.selection import (
    ScenedCandidate,
    chunk_for_tournament,
    dedup_by_score,
    diverse_preselection,
    exact_twins,
    is_duplicate,
    select_day,
)


@dataclass
class Photo:
    path: Path
    key: str
    source: Path | None = None  # file actually decoded (a DxO twin), None = path itself
    time: datetime | None = None
    emb: np.ndarray | None = None
    sig: np.ndarray | None = None
    dino: np.ndarray | None = None
    subject_dino: np.ndarray | None = None
    subject_dino_checked: bool = False
    subject_boxes: list[tuple] = field(default_factory=list)
    subject_boxes_checked: bool = False
    subject_clip: np.ndarray | None = None
    interaction_clip: np.ndarray | None = None
    interaction_dino: np.ndarray | None = None
    burst_features_checked: bool = False
    burst: int = -1
    burst_signals: dict[str, float] = field(default_factory=dict)
    digest: str = ""
    box: tuple | None = None
    subject_sharpness: float | None = None
    detail_focus: float | None = None
    detail_focus_checked: bool = False
    focus_rank: float = 0.0
    refined_focus_adjustment: float = 0.0
    focus_refinement: str = ""
    person: float = 0.0  # largest detected person box, fraction of frame
    subject: float = 0.0  # largest detected person/animal box, fraction of frame
    edge: bool = False  # that box touches the frame border
    judge_reason: str = ""
    judge_label: str = ""  # subject as named by the judge; species folders follow it when present
    award_rank: int = 0
    award_assessment: AwardAssessment | None = None
    shortlisted: bool = False
    iso: int | None = None
    edit_score: int = 0          # 0..100 heuristic: how much a RAW edit would gain
    edit_band: str = ""          # low / medium / high (judge's word when judged, else from edit_score)
    edit_why: str = ""
    preset: str = ""
    sharpness: float = 0.0
    clip_low: float = 0.0
    clip_high: float = 0.0
    mean: float = 0.0
    aesthetic: float = 0.0
    composition: dict[str, float | None] = field(default_factory=dict)
    editorial: dict[str, float] = field(default_factory=dict)
    composition_bonus: float = 0.0
    editorial_bonus: float = 0.0
    label: str = ""
    label_prob: float = 0.0
    people_prob: float | None = None  # CLIP mass on people labels; gates the detector override
    scene: int = -1
    moment: int = -1
    moment_novelty: float = 0.0
    preselection_reason: str = ""
    score: float = 0.0
    reject: str = ""
    selected: bool = False
    decision: str = ""
    auto_selected: bool | None = None
    auto_decision: str = ""
    manual_decision: str = ""
    manual_reason: str = ""
    review_status: str = ""
    review_revision: int = 0

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
    p.add_argument("--mode", default="standard", choices=["standard", "award-roll"],
                   help="standard: highlights per day/subject; award-roll: one wildlife portfolio across all dates "
                        "(requires --judge, defaults to a bounded local shortlist)")
    p.add_argument("--roll-size", type=int, default=12, choices=range(10, 16),
                   help="Target and maximum photos for award-roll (10–15); may return fewer if quality warrants")
    p.add_argument("--award-candidates", type=int, default=480,
                   help="Maximum distinct judge candidates across the collection in award-roll preselection "
                        "(default 480); full coverage explicitly bypasses this cap")
    p.add_argument("--award-plan", action="store_true",
                   help="With award-roll: analyze locally and report the candidate plan and comparison bounds "
                        "without calling the judge or exporting photos")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--move", action="store_true", help="Move picks into Highlights (default: dry run)")
    g.add_argument("--copy", action="store_true", help="Copy picks into Highlights instead of moving")
    p.add_argument("--highlights", help="Output subfolder (Highlights, or AwardRoll in award-roll mode)")
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
    p.add_argument("--dup-reframe-sim", type=float, default=0.98,
                   help="Strong full-image DINO match allowing reframed local duplicates when subject DINO "
                        "also agrees within a scene (0 disables)")
    p.add_argument("--min-score", type=float, default=float("-inf"),
                   help="Optional relative score cutoff (disabled by default; quality checks still apply)")
    p.add_argument("--subject-weight", type=float, default=0.4, help="Bonus weight for a large detected subject")
    p.add_argument("--judge", action="store_true",
                   help="Let a vision model choose the final picks from each group's shortlist")
    p.add_argument("--judge-provider", default="openai", choices=["openai", "anthropic"], help="API for --judge")
    p.add_argument("--judge-model", help="Model for --judge (default: gpt-5.6-sol / claude-opus-5)")
    p.add_argument("--judge-coverage", choices=["preselect", "full"],
                   help="preselect (default): local shortlist; full: every distinct eligible file, "
                        "bypassing the award-roll candidate cap")
    p.add_argument("--preselect", type=float, default=3.0,
                   help="Preselection size as a multiple of the group's budget (at least --preselect-min frames, "
                        "and at least --preselect-share of the group)")
    p.add_argument("--preselect-min", type=int, default=12, help="Minimum preselection size per group")
    p.add_argument("--moment-gap-seconds", type=float, default=8.0,
                   help="Start a new moment inside a scene after this capture gap")
    p.add_argument("--moment-sim", type=float, default=0.92,
                   help="Whole-image or subject DINO similarity below which a new moment starts")
    p.add_argument("--subject-embeddings", action=argparse.BooleanOptionalAction, default=True,
                   help="Compare detected subject crops locally with DINO (cached)")
    p.add_argument("--burst-alternatives", action=argparse.BooleanOptionalAction, default=False,
                   help="Experimental: reserve shortlist capacity for burst poses, expressions and interactions; "
                        "adds cached local crop analysis when preselecting for the judge (off by default)")
    p.add_argument("--preselect-explore", type=int, default=2,
                   help="Maximum shortlist slots reserved for unusual/uncertain alternatives (0 disables)")
    p.add_argument("--focus-refine-per-group", type=int, default=12,
                   help="Maximum frames per day/subject for extra local focus checks of close winners (0 disables)")
    p.add_argument("--preselect-share", type=float, default=0.5,
                   help="Minimum share of each distinct eligible day/subject group to shortlist "
                        "(default 0.5, rounded up); 0 opts into a smaller, cheaper shortlist with more risk "
                        "of missing exceptional frames")
    p.add_argument("--include", default="",
                   help="Frames that must be picked regardless of scores or judge: comma-separated stems or file "
                        "names, or @file with one per line (e.g. --include P9051472,P9050886)")
    p.add_argument("--judge-chunk", type=int, default=24,
                   help="Frames per initial request; winner comparisons combine pools up to this size "
                        "or twice the group budget, whichever is larger")
    p.add_argument("--taste-dir", help="Folder with photos you love; frames that resemble them get a score bonus")
    p.add_argument("--taste-weight", type=float, default=0.5, help="Max bonus from --taste-dir")
    p.add_argument("--judge-species", action=argparse.BooleanOptionalAction, default=True,
                   help="With --judge: let the judge name the subject of each pick; species folders follow "
                        "its answer instead of the local CLIP label (about one request per 12 picks)")
    p.add_argument("--judge-hint", default="",
                   help="One or two sentences about this shoot for the judge, e.g. what counts as a keeper")
    p.add_argument("--judge-detail", default="high", choices=["high", "low"],
                   help="Full-image detail sent to the OpenAI judge; low reduces overview resolution and tokens")
    p.add_argument("--judge-crops", default="single", choices=["single", "multi", "none"],
                   help="Subject details per frame: single (default) native crop; multi adds an overview and "
                        "several native windows; none sends only the full image")
    p.add_argument("--key-file", help="File holding the API key (a .env or YAML line), instead of the environment")
    p.add_argument("--judge-base-url",
                   help="OpenAI-compatible server for a local judge, e.g. http://localhost:11434/v1 (Ollama); "
                        "requires --judge-model; no key or upload needed")
    p.add_argument("--xmp", choices=["picks", "all"],
                   help="Write XMP sidecars next to the originals: rating 5 + keywords for picks (and, with "
                        "'all', rating 3 for judged-but-not-picked, 1 for rejected frames); existing sidecars "
                        "are left alone unless --xmp-overwrite")
    p.add_argument("--xmp-overwrite", action="store_true", help="Replace existing .xmp sidecars")
    p.add_argument("--raw-cull", action="store_true",
                   help="List which RAW files (same stem next to the JPEG or in RAW/) are worth keeping: those of "
                        "picks whose edit benefit is at least --raw-keep-benefit. Writes raw_keep.txt and "
                        "raw_cull.txt into the folder")
    p.add_argument("--raw-keep-benefit", default="low", choices=["low", "medium", "high"],
                   help="Keep the RAW of a pick only if its edit benefit is at least this")
    p.add_argument("--raw-cull-move", action="store_true",
                   help="With --raw-cull: move the RAWs not worth keeping into a _DELETE_ME_raw folder (never deletes)")
    p.add_argument("--detector", default="yolov8m.pt", help="YOLOv8 weights for subject detection (n/s/m/l)")
    p.add_argument("--aesthetic-weight", type=float, default=0.6, help="Weight of aesthetics vs sharpness (0..1)")
    p.add_argument("--composition-weight", type=float, default=0.2,
                   help="Maximum local bonus for thirds, golden-ratio or central placement (0 disables)")
    p.add_argument("--editorial-weight", type=float, default=0.15,
                   help="Maximum local CLIP preference bonus for light, composition, subject and moment (0 disables)")
    p.add_argument("--blur-ratio", type=float, default=0.15,
                   help="Without the judge, reject if sharpness < ratio * known scene median")
    p.add_argument("--blur-floor", type=float, default=20.0,
                   help="Without the judge, reject if sharpness is below this absolute value")
    p.add_argument("--labels", help="Text file with one subject label per line (overrides defaults)")
    p.add_argument("--label-mode", default="scene", choices=["scene", "image"],
                   help="scene: one subject label per burst (steady for walking safaris); image: label every frame "
                        "on its own (use on a boat or anywhere the background never changes)")
    p.add_argument("--person-agree", type=float, default=0.2,
                   help="Detector person override needs at least this CLIP probability mass on people labels "
                        "(seals and other animals fool the detector; 0 trusts the detector alone)")
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
    p.add_argument("--report", help="CSV report path relative to folder "
                   "(fotosort_report.csv, or award_roll_report.csv in award-roll mode)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4, help="Decoder threads")
    p.add_argument("--no-cache", action="store_true", help="Ignore and do not write the feature cache")
    p.add_argument("--limit", type=int, help="Only process the first N files (for testing)")
    p.add_argument("--apply-report", action="store_true",
                   help="Skip analysis: take the picks (selected=1) from the existing report and copy/move/enhance "
                        "them. Edit the CSV first to override the tool's choices.")
    args = p.parse_args(argv)
    if args.highlights is None:
        args.highlights = "AwardRoll" if args.mode == "award-roll" else "Highlights"
    if args.report is None:
        args.report = ("award_roll_plan.csv" if args.award_plan else
                       "award_roll_report.csv" if args.mode == "award-roll" else "fotosort_report.csv")
    if args.judge_coverage is None:
        args.judge_coverage = "preselect"
    if args.award_plan and (args.mode != "award-roll" or args.apply_report or args.copy or args.move or args.enhance):
        p.error("--award-plan requires --mode award-roll and cannot be combined with "
                "--apply-report, --copy, --move or --enhance")
    if args.mode == "award-roll" and not args.apply_report:
        if not args.judge and not args.award_plan:
            p.error("--mode award-roll requires --judge for visual wildlife and exceptional-moment decisions")
        if args.include:
            p.error("--include cannot bypass the award-roll species rules and total cap; "
                    "edit the resulting CSV and use --apply-report for manual overrides")
        if args.judge_coverage == "preselect" and args.award_candidates < args.roll_size:
            p.error("--award-candidates must be at least --roll-size")
    for name in ("batch_size", "workers", "judge_chunk", "max_per_group", "preselect_min", "award_candidates"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.preselect_share <= 1 or args.preselect <= 0:
        p.error("--preselect-share must be between 0 and 1; --preselect must be positive")
    for name in ("aesthetic_weight", "composition_weight", "editorial_weight", "dup_reframe_sim"):
        if not 0 <= getattr(args, name) <= 1:
            p.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if not np.isfinite(args.moment_gap_seconds) or args.moment_gap_seconds <= 0:
        p.error("--moment-gap-seconds must be finite and positive")
    if not 0 <= args.moment_sim <= 1:
        p.error("--moment-sim must be between 0 and 1")
    if args.preselect_explore < 0 or args.focus_refine_per_group < 0:
        p.error("--preselect-explore and --focus-refine-per-group must be nonnegative")
    return args


def _decode(photo: Photo, prepare):
    """Returns (photo, sharpness, exposure, tensor, small image) or (photo, None, error, None, None)."""
    try:
        img = load_small(str(photo.source or photo.path))
        gray = to_gray(img)
        ex = exposure(gray)
        photo.sig = signature(img)
        photo.digest = file_digest(str(photo.source or photo.path))
        return photo, sharpness(gray), ex, prepare(img), img
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
            ph.digest, ph.box, ph.subject_sharpness = c["digest"], c["box"], c["subject_sharpness"]
            ph.subject_dino = c["subject_dino"] if args.subject_embeddings else None
            ph.subject_dino_checked = c["subject_dino_checked"]
            ph.detail_focus, ph.detail_focus_checked = c["detail_focus"], c["detail_focus_checked"]
            for name in ("subject_boxes", "subject_boxes_checked", "subject_clip", "interaction_clip",
                         "interaction_dino", "burst_features_checked"):
                setattr(ph, name, c[name])
        else:
            todo.append(ph)
    print(f"{len(photos) - len(todo)} cached, {len(todo)} to analyse")

    from fotosort.embed import Embedder  # heavy import, only when needed

    print("Loading CLIP ViT-L/14 + aesthetic predictor ...")
    emb = Embedder()
    print(f"Running on {emb.device}")

    detector = dino = None

    def embed_subjects(batch, smalls):
        cropped, owners = [], []
        for ph, small in zip(batch, smalls, strict=True):
            crop = subject_embedding_crop(small, ph.box)
            ph.subject_dino_checked = True
            if crop is not None:
                cropped.append(crop)
                owners.append(ph)
        if cropped:
            for ph, vec in zip(owners, dino.embed_batch(cropped), strict=True):
                ph.subject_dino = vec

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
                for p_, im_, d_ in zip(batch, smalls, detector.detect(smalls), strict=True):
                    p_.person, p_.subject, p_.edge = d_["person"], d_["subject"], d_["edge"]
                    p_.box = d_["box"]
                    p_.subject_boxes = valid_boxes(d_.get("boxes", [p_.box] if p_.box is not None else []))
                    p_.subject_boxes_checked = True
                    if p_.box is not None:
                        p_.subject_sharpness = subject_focus(im_, p_.box)
            if args.subject_embeddings:
                embed_subjects(batch, smalls)

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

    # Upgrade v6 or previously disabled subject features without repeating CLIP,
    # detection or whole-frame DINO. Keep decoding and GPU work batch bounded.
    backfill = [p for p in photos if p.emb is not None and not p.subject_dino_checked
                and args.subject_embeddings]
    if backfill:
        print(f"Adding local subject comparisons for {len(backfill)} cached frames ...")
        if dino is None and any(p.box is not None for p in backfill):
            from fotosort.embed import DinoEmbedder

            dino = DinoEmbedder(device=emb.device)
        for start in range(0, len(backfill), args.batch_size):
            batch, smalls = [], []
            for ph in backfill[start:start + args.batch_size]:
                if ph.box is None:
                    ph.subject_dino_checked = True
                    continue
                try:
                    smalls.append(load_small(str(ph.source or ph.path)))
                    batch.append(ph)
                except Exception as exc:
                    tqdm.write(f"Subject comparison unavailable for {ph.path.name}: {exc}")
            embed_subjects(batch, smalls)
    burst_updates = []
    if (args.judge or args.award_plan) and args.judge_coverage == "preselect" and args.burst_alternatives:
        burst_updates = enrich_burst_features(photos, args, emb, detector, dino)
    if not args.no_cache and (todo or backfill or burst_updates):
        for ph in todo + backfill + burst_updates:
            if ph.emb is not None:
                cached[ph.key] = _feature_entry(ph)
        cache_mod.save(root, cached, det_name)

    readable = [ph for ph in photos if ph.emb is not None]
    if readable:
        all_emb = np.stack([ph.emb for ph in readable])
        for ph, a in zip(readable, emb.aesthetic(all_emb), strict=True):
            ph.aesthetic = float(a)
        if args.editorial_weight > 0:
            prompts = [prompt for pair in EDITORIAL_PROMPTS.values() for prompt in pair]
            scores = editorial_scores(all_emb, emb.text_embeddings(prompts, template="{}"))
            for i, ph in enumerate(readable):
                ph.editorial = {name: float(values[i]) for name, values in scores.items()}
    return emb


def enrich_burst_features(photos, args, emb, detector=None, dino=None):
    """Up to two extra CLIP views and one DINO view per image, all local/cached."""
    pending = [ph for ph in photos if ph.emb is not None and not ph.burst_features_checked]
    if pending:
        print(f"Analysing local burst alternatives for {len(pending)} frames ...")
    if any(not ph.subject_boxes_checked for ph in pending) and args.person_area > 0 and detector is None:
        from fotosort.embed import SubjectDetector

        detector = SubjectDetector(device=emb.device, weights=args.detector)
    changed = []
    for start in tqdm(range(0, len(pending), args.batch_size), desc="Burst detail", unit="batch",
                      disable=not pending, leave=False):
        batch, images = [], []
        for ph in pending[start:start + args.batch_size]:
            try:
                images.append(load_small(str(ph.source or ph.path)))
                batch.append(ph)
            except Exception as exc:
                tqdm.write(f"Burst detail unavailable for {ph.path.name}: {exc}")
        missing = [i for i, ph in enumerate(batch) if not ph.subject_boxes_checked]
        if missing and detector is not None:
            for i, detection in zip(missing, detector.detect([images[i] for i in missing]), strict=True):
                box = detection["box"]
                batch[i].subject_boxes = valid_boxes(detection.get("boxes", [box] if box is not None else []))
                batch[i].subject_boxes_checked = True
        tensors, owners, contexts, context_owners = [], [], [], []
        for ph, image in zip(batch, images, strict=True):
            for name, crop in local_burst_views(image, ph.box, ph.subject_boxes).items():
                tensors.append(emb.prepare(crop))
                owners.append((ph, name))
                if name == "interaction":
                    contexts.append(crop)
                    context_owners.append(ph)
        # Keep GPU batches bounded even though each source can contribute two views.
        for j in range(0, len(tensors), args.batch_size):
            for (ph, name), vector in zip(owners[j:j + args.batch_size],
                                          emb.embed_batch(tensors[j:j + args.batch_size]), strict=True):
                setattr(ph, name + "_clip", vector)
        if contexts and dino is None:
            from fotosort.embed import DinoEmbedder

            dino = DinoEmbedder(device=emb.device)
        if contexts:
            for ph, vector in zip(context_owners, dino.embed_batch(contexts), strict=True):
                ph.interaction_dino = vector
        for ph in batch:
            ph.burst_features_checked = True
            changed.append(ph)
    texts = emb.text_embeddings([prompt for pair in BURST_PROMPTS.values() for prompt in pair], template="{}")
    for ph in photos:
        if ph.emb is not None:
            ph.burst_signals = burst_scores(ph.emb, ph.subject_clip, ph.interaction_clip, texts)
    return changed


def _feature_entry(ph: Photo) -> dict:
    names = ("emb", "sig", "dino", "subject_dino", "subject_dino_checked", "person", "subject", "edge",
             "sharpness", "clip_low", "clip_high", "mean", "digest", "box", "subject_sharpness",
             "detail_focus", "detail_focus_checked", "subject_boxes", "subject_boxes_checked",
             "subject_clip", "interaction_clip", "interaction_dino", "burst_features_checked")
    return {name: getattr(ph, name) for name in names}


def _z(x: np.ndarray) -> np.ndarray:
    if not len(x):
        return np.zeros_like(x)
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
    """Rank within each day's subject group, using subject detail when available.

    Aesthetic and focus ranks are relative preferences, not absolute usability
    tests. Other days/species must not make an existing keeper fall below a
    cutoff. The visual judge gets to assess difficult light and low texture.
    """
    skip = {s.strip() for s in args.skip_labels.split(",") if s.strip()}
    groups = defaultdict(list)
    scenes = defaultdict(list)
    for ph in photos:
        ph.composition = composition_metrics(ph.box)
        groups[(ph.day, ph.label)].append(ph)
        scenes[(ph.day, ph.label, ph.scene)].append(ph)
    for group in groups.values():
        aest = _z(np.array([ph.aesthetic for ph in group]))
        editorial_peers = [ph for ph in group if "editorial_score" in ph.editorial]
        editorial_ranks = {id(ph): float(rank) for ph, rank in
                           zip(editorial_peers, _z(np.array([p.editorial["editorial_score"]
                                                            for p in editorial_peers])), strict=True)}
        # Global and subject-region Laplacians have different scales. Compare
        # only like measurements, then combine their relative ranks.
        focus_ranks = {}
        for has_subject in (False, True):
            peers = [p for p in group if (p.subject_sharpness is not None) == has_subject]
            values = [p.subject_sharpness if has_subject else p.sharpness for p in peers]
            for p, rank in zip(peers, _z(np.log1p(values)), strict=True):
                focus_ranks[id(p)] = float(rank)
        for i, ph in enumerate(group):
            ph.focus_rank = focus_ranks[id(ph)]
            ph.refined_focus_adjustment = 0.0
            penalty = 3.0 * max(0.0, ph.clip_high - 0.02) + 2.0 * max(0.0, ph.clip_low - 0.10)
            ph.composition_bonus = args.composition_weight * (ph.composition["composition_score"] or 0.0)
            ph.editorial_bonus = args.editorial_weight * float(np.clip(editorial_ranks.get(id(ph), 0) / 2, 0, 1))
            ph.score = (args.aesthetic_weight * float(aest[i])
                        + (1 - args.aesthetic_weight) * focus_ranks[id(ph)]
                        - penalty + subject_term(ph, args.subject_weight)
                        + ph.composition_bonus + ph.editorial_bonus)
            ph.reject = ""
            if ph.label in skip:
                ph.reject = "label"
            elif not args.judge and args.mode != "award-roll":
                peers = scenes[(ph.day, ph.label, ph.scene)]
                # Relative blur checks only compare a known burst, never a
                # smooth bird with the entire day's textured landscapes.
                median = np.median([p.sharpness for p in peers]) if ph.scene >= 0 and len(peers) >= 3 else 0
                if ph.sharpness < args.blur_floor or ph.sharpness < args.blur_ratio * median:
                    ph.reject = "blurry"
                elif ph.clip_high > 0.4 or ph.clip_low > 0.6:
                    ph.reject = "exposure"


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
    people_idx = [i for i, lab in enumerate(labels) if is_people_label(lab)]
    if photos and people_idx:
        for p, mass in zip(photos, people_mass(np.stack([p.emb for p in photos]), text_emb, people_idx), strict=True):
            p.people_prob = float(mass)
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
    apply_person_override(photos, args.person_area, args.person_agree)


def assign_moments(photos: list[Photo], args) -> None:
    """Find pose/composition changes within scenes, without expanding budgets."""
    groups = defaultdict(list)
    for ph in photos:
        groups[(ph.day, ph.label)].append(ph)
    offset = burst_offset = 0
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda p: (p.time or datetime.max, str(p.path)))
        ids, novelty = cluster_moments(
            [p.time.timestamp() if p.time else None for p in group], [p.scene for p in group],
            [p.dino if p.dino is not None else p.emb for p in group], [p.subject_dino for p in group],
            args.moment_gap_seconds, args.moment_sim)
        for ph, moment, change in zip(group, ids, novelty, strict=True):
            ph.moment, ph.moment_novelty = moment + offset, change
        offset += max(ids, default=-1) + 1
        bursts = cluster_bursts(
            [p.time.timestamp() if p.time else None for p in group], [p.scene for p in group],
            [p.dino if p.dino is not None else p.emb for p in group]) if args.burst_alternatives else [-1] * len(group)
        for ph, burst in zip(group, bursts, strict=True):
            ph.burst = burst + burst_offset if burst >= 0 else -1
        burst_offset += max(bursts, default=-1) + 1


def refine_close_focus(photos: list[Photo], root: Path, args) -> None:
    """Spend a bounded amount of local work resolving plausible moment winners.

    Compare at most three alternatives in a moment, only when overall scores
    are within one point and scores excluding focus within half a point. These
    are scheduling heuristics, not rejection criteria. Native subject detail
    replaces the bounded focus preference only within a successfully read set.
    """
    if args.focus_refine_per_group < 2 or args.aesthetic_weight == 1:
        return
    groups = defaultdict(lambda: defaultdict(list))
    for ph in photos:
        if not ph.reject and ph.box is not None and ph.moment >= 0:
            groups[(ph.day, ph.label)][ph.moment].append(ph)
    weight = 1 - args.aesthetic_weight
    changed = []
    for moments in tqdm(groups.values(), desc="Focus refinement", unit="group", leave=False):
        remaining = args.focus_refine_per_group
        for peers in sorted(moments.values(), key=lambda ps: (-max(p.score for p in ps), min(str(p.path) for p in ps))):
            if remaining < 2:
                break
            ranked, seen = [], set()
            for ph in sorted(peers, key=lambda p: (-p.score, str(p.path))):
                if ph.digest and ph.digest in seen:
                    continue
                ranked.append(ph)
                if ph.digest:
                    seen.add(ph.digest)
            leader = ranked[0]
            alternatives = [p for p in ranked[1:] if leader.score - p.score <= 1.0
                            and abs((leader.score - weight * leader.focus_rank)
                                    - (p.score - weight * p.focus_rank)) <= 0.5]
            if not alternatives:
                continue
            compared = [leader, *alternatives[:min(2, remaining - 1)]]
            remaining -= len(compared)
            for ph in compared:
                if not ph.detail_focus_checked:
                    try:
                        ph.detail_focus = detail_focus(str(ph.source or ph.path), ph.box)
                        ph.detail_focus_checked = True
                        changed.append(ph)
                    except Exception as exc:
                        ph.focus_refinement = f"Detail unavailable: {exc}"
                        continue
                ph.focus_refinement = ("Subject too small for detail comparison"
                                       if ph.detail_focus is None else "Detail read")
            if not all(p.detail_focus is not None for p in compared):
                continue  # never compare a native/detail score with a preview score
            ranks = _z(np.log1p([p.detail_focus for p in compared]))
            for ph, rank in zip(compared, ranks, strict=True):
                new_rank = float(np.clip(rank, -1, 1))
                ph.refined_focus_adjustment = weight * (new_rank - float(np.clip(ph.focus_rank, -1, 1)))
                ph.score += ph.refined_focus_adjustment
                ph.focus_rank = new_rank
                ph.focus_refinement = "Compared within moment at 512 px subject scale"
    if changed:
        print(f"Refined subject focus locally for {len(changed)} frames")
    if changed and not args.no_cache:
        det_name = args.detector if args.person_area > 0 else "none"
        cached = cache_mod.load(root, det_name)
        for ph in changed:
            if ph.key not in cached and ph.emb is not None:
                cached[ph.key] = _feature_entry(ph)
            if ph.key in cached:
                cached[ph.key].update(detail_focus=ph.detail_focus, detail_focus_checked=ph.detail_focus_checked)
        cache_mod.save(root, cached, det_name)


def people_mass(emb: np.ndarray, text_emb: np.ndarray, people_idx: list[int]) -> np.ndarray:
    """Per-image probability mass CLIP puts on people-like labels."""
    logits = 100.0 * emb @ text_emb.T
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return p[:, people_idx].sum(axis=1)


def apply_person_override(photos: list[Photo], min_area: float, min_agree: float = 0.0) -> None:
    """A clearly visible person beats whatever CLIP thought the scene was, unless
    CLIP sees almost no person at all: swimming seals and other animals trigger
    the detector's person class, so a small box needs CLIP agreement. A person
    filling half the frame is trusted regardless."""
    if min_area <= 0:
        return
    for p in photos:
        if p.person < min_area:
            continue
        agrees = p.people_prob is None or min_agree <= 0 or p.people_prob >= min_agree or p.person >= 0.5
        if agrees:
            p.label, p.label_prob = "people", max(p.label_prob, p.person)


def forced_includes(photos: list[Photo], spec: str) -> list[Photo]:
    """Resolve --include (stems, file names, or @file) to photos; they are never
    rejected and always end up selected."""
    if not spec:
        return []
    if spec.startswith("@"):
        names = [ln.strip() for ln in Path(spec[1:]).expanduser().read_text().splitlines() if ln.strip()]
    else:
        names = [n.strip() for n in spec.split(",") if n.strip()]
    wanted = {Path(n).stem.lower() for n in names}
    found = [ph for ph in photos if ph.path.stem.lower() in wanted]
    for ph in found:
        ph.reject = ""
    missing = wanted - {ph.path.stem.lower() for ph in found}
    if missing:
        print(f"--include: not found in this folder: {', '.join(sorted(missing))}")
    return found


def apply_edit_verdict(ph: Photo, verdict, i: str) -> None:
    e = (verdict.edits or {}).get(i)
    if not e:
        return
    if e.get("benefit"):
        ph.edit_band, ph.edit_why = e["benefit"], e.get("why", "")
    if e.get("preset"):
        ph.preset = e["preset"]


def editing_advice(photos: list[Photo], emb, args) -> None:
    """Edit benefit for every frame from its statistics; for picks without a
    judge verdict also a preset from the detected style."""
    from fotosort.enhance import STYLE_PROMPTS, detect_style, image_stats

    for ph in photos:
        if ph.emb is None:
            continue
        ph.edit_score = edit_benefit(ph.clip_high, ph.clip_low, ph.mean, None, ph.iso)
        if not ph.edit_band:
            ph.edit_band = benefit_band(ph.edit_score)
    need = [ph for ph in photos if ph.selected and not ph.preset]
    if not need or emb is None:
        return
    style_emb = emb.text_embeddings(list(STYLE_PROMPTS.values()), template="{}")
    for ph in need:
        try:
            stats = image_stats(np.asarray(load_small(str(ph.source or ph.path), 512), dtype=np.float32))
        except OSError:
            continue
        style = "portrait" if ph.person >= args.person_area > 0 else detect_style(ph.emb, style_emb, stats)
        ph.preset = STYLE_TO_PRESET.get(style, "Natural")


def raw_cull(photos: list[Photo], root: Path, args) -> None:
    """Which RAW files to keep: those of picks whose edit benefit reaches
    --raw-keep-benefit. Writes raw_keep.txt / raw_cull.txt; with
    --raw-cull-move the culled RAWs go into _DELETE_ME_raw (nothing is deleted)."""
    order = {"low": 0, "medium": 1, "high": 2}
    keep, cull, missing = set(), set(), 0
    for ph in photos:
        raw = raw_sibling(ph.path)
        if raw is None:
            missing += 1
            continue
        raw = raw.resolve()
        if ph.selected and order.get(ph.edit_band, 0) >= order[args.raw_keep_benefit]:
            keep.add(raw)
        else:
            cull.add(raw)
    cull.difference_update(keep)  # one qualifying representation protects the RAW
    keep, cull = sorted(keep), sorted(cull)
    (root / "raw_keep.txt").write_text("\n".join(str(p) for p in keep) + "\n")
    (root / "raw_cull.txt").write_text("\n".join(str(p) for p in cull) + "\n")
    size = sum(p.stat().st_size for p in cull if p.exists()) / 1e9
    print(f"RAW cull: keep {len(keep)}, cull {len(cull)} ({size:.1f} GB)" +
          (f", {missing} frames have no RAW" if missing else "") + f"; lists in {root}")
    if args.raw_cull_move and cull:
        dest = root / "_DELETE_ME_raw"
        dest.mkdir(exist_ok=True)
        moved = 0
        for p in cull:
            if p.exists():
                shutil.move(str(p), str(unique_dest(dest, p, p.parent)))
                moved += 1
        print(f"Moved {moved} RAW files into {dest}; delete that folder when you are sure")


def judge_with_retry(judge, paths, label, day, k, boxes, final=False, attempts: int = 3):
    """A transient network error must not decide a group's picks: retry with a pause."""
    import time

    verdict = None
    for attempt in range(attempts):
        verdict = judge.judge(paths, label, day, k, boxes, final=final)
        if not verdict.error or "declined" in verdict.error or "vanished" in verdict.error:
            return verdict
        if attempt < attempts - 1:
            tqdm.write(f"Judge error for {day} {label} ({verdict.error[:60]}); retrying in {10 * (attempt + 1)}s")
            time.sleep(10 * (attempt + 1))
    return verdict


def judge_boxes(photos: list[Photo], judge) -> dict[str, tuple | None]:
    """Reuse cached subject boxes in the same orientation as the decoded source."""
    return {str(p.path): p.box for p in photos}


def pick_budget(n_photos: int, base: int, extra_per: int) -> int:
    """Base picks plus one per `extra_per` photos, capped at 3x base: a 450-frame
    afternoon of the same subject deserves more keepers than a 6-frame one."""
    extra = n_photos // extra_per if extra_per > 0 else 0
    return min(base + extra, 3 * base)


def _candidate(p: Photo) -> ScenedCandidate:
    return ScenedCandidate(str(p.path), p.score, p.label, p.emb, p.sig, p.dino, p.scene, digest=p.digest,
                           aesthetic=p.aesthetic, composition=p.composition_bonus, editorial=p.editorial_bonus,
                           focus=p.focus_rank, subject_dino=p.subject_dino, moment=p.moment,
                           distinctiveness=p.moment_novelty, uncertainty=1 - p.label_prob, burst=p.burst,
                           capture_time=p.time.timestamp() if p.time else None,
                           burst_signals=p.burst_signals if p.burst >= 0 else {},
                           interaction_dino=p.interaction_dino if p.burst >= 0 else None)


def _judge_round(ids: list[str], by_path: dict[str, Photo], judge, label: str, day: str,
                 k: int, args, final: bool = False) -> list[str]:
    ids = [i for i in ids if (by_path[i].source or by_path[i].path).exists()]
    if not ids:
        return []
    k = min(k, len(ids))
    for i in ids:
        by_path[i].shortlisted = True
    verdict = judge_with_retry(judge, [Path(i) for i in ids], label, day, k,
                               judge_boxes([by_path[i] for i in ids], judge), final=final)
    if verdict.error:
        if args.mode == "award-roll":
            raise SystemExit(f"Award-roll judging failed ({verdict.error}); no portfolio was exported. "
                             "Check the judge and retry; relative local scores cannot replace this comparison.")
        tqdm.write(f"Judge failed for {day} {label} ({verdict.error}); using scores instead")
        judge.failures = getattr(judge, "failures", 0) + 1
        if judge.failures >= 3:
            raise SystemExit("The judge failed three times in a row; check the API key, model name "
                             "and network, or run without --judge.")
        picks = dedup_by_score([_candidate(by_path[i]) for i in ids], k, args.dup_sim, args.dup_pixel,
                              args.dup_reframe_sim)
        verdict = Verdict(picks, {i: "(judge failed, by score)" for i in picks})
    else:
        judge.failures = 0
    # Validate even custom/local judge implementations at the pipeline boundary.
    picked = list(dict.fromkeys(i for i in verdict.picks if i in ids))
    dropped = {}
    if args.mode == "award-roll":
        assessments = verdict.awards or {}
        if any(i not in assessments for i in picked):
            raise SystemExit("Award-roll judge returned picks without wildlife/species/moment evidence; "
                             "no portfolio was exported.")
        for i in picked:
            by_path[i].award_assessment = assessments[i]
            by_path[i].judge_label = assessments[i].species
        picked, dropped = checked_award_picks(picked, assessments, k, curate=judge.award_final)
    else:
        picked = picked[:k]
    for i in ids:
        if i in picked:
            by_path[i].judge_reason = verdict.reasons.get(i) or by_path[i].judge_reason
            apply_edit_verdict(by_path[i], verdict, i)
        else:
            by_path[i].judge_reason = dropped.get(i) or (
                "Not selected in final comparison" if final else "Not selected by judge")
    return picked


def _tournament(rounds: list[list[str]], by_path: dict[str, Photo], judge, label: str,
                day: str, k: int, args, final: bool = False) -> list[str]:
    """Every chunk may advance the full budget; combine winner pools that fit.

    Under a consistent ranking this preserves the global top k even if every
    exceptional frame is concentrated in one chunk. Merge requests hold at
    most max(chunk, 2*k) images; fewer merge rounds mean fewer repeat uploads.
    Every stage strictly reduces the number of pools.
    """
    pools = [_judge_round(ids, by_path, judge, label, day, k, args, final=final) for ids in rounds]
    pools = [pool for pool in pools if pool]
    request_limit = max(args.judge_chunk, 2 * k)
    while len(pools) > 1:
        batches, batch, size = [], [], 0
        for pool in pools:
            if batch and size + len(pool) > request_limit:
                batches.append(batch)
                batch, size = [], 0
            batch.append(pool)
            size += len(pool)
        if batch:
            batches.append(batch)
        merged = [batch[0] if len(batch) == 1 else
                  _judge_round([i for pool in batch for i in pool], by_path, judge,
                               label, day, k, args, final=True) for batch in batches]
        pools = [pool for pool in merged if pool]
    return pools[0] if pools else []


def _reconcile_day(chosen: set[str], by_path: dict[str, Photo], judge, day: str, args) -> set[str]:
    """Compare suspected duplicates across labels without a thumbnail veto.

    Exact copies collapse directly. For visually similar files the judge may
    retain both when they depict different worthwhile moments.
    """
    cands, twins = exact_twins([_candidate(by_path[i]) for i in sorted(chosen)])
    for dropped, kept in twins.items():
        by_path[dropped].judge_reason = f"= identical copy of {Path(kept).name}"
        chosen.discard(dropped)
    edges = {c.id: set() for c in cands}
    for i, c in enumerate(cands):
        for other in cands[i + 1:]:
            if c.bucket != other.bucket and is_duplicate(c, [other], args.dup_sim, args.dup_pixel,
                                                        args.dup_reframe_sim):
                edges[c.id].add(other.id)
                edges[other.id].add(c.id)
    visited = set()
    for start in edges:
        if start in visited or not edges[start]:
            continue
        pending, component = [start], []
        while pending:
            i = pending.pop()
            if i in visited:
                continue
            visited.add(i)
            component.append(i)
            pending.extend(sorted(edges[i] - visited))
        component.sort()
        rounds = [component[i:i + args.judge_chunk] for i in range(0, len(component), args.judge_chunk)]
        winners = _tournament(rounds, by_path, judge, "mixed subjects; check for duplicate moments",
                              day, len(component), args, final=True)
        chosen.difference_update(component)
        chosen.update(winners)
    return chosen


def select_award_roll(photos: list[Photo], args, judge) -> dict[tuple[str, str], list[Photo]]:
    """Nominate broadly, then curate one roll across the entire collection.

    Local scores only order candidates and optional within-group preselection.
    Each initial batch can advance the full finalist budget (twice the roll
    size); the final visual comparison applies species exceptions and the cap.
    """
    if judge is None and not args.award_plan:
        raise ValueError("award-roll requires a visual judge")
    buckets = defaultdict(list)
    by_path = {str(p.path): p for p in photos}
    for ph in photos:
        ph.selected = ph.shortlisted = False
        ph.award_rank, ph.award_assessment = 0, None
        ph.judge_reason = ph.judge_label = ph.preselection_reason = ""
        ph.decision = f"Rejected: {ph.reject}" if ph.reject else "Outside judge shortlist"
        buckets[(ph.day, ph.label)].append(ph)
    if judge is not None:
        judge.sources = {str(p.path): p.source or p.path for p in photos}
    candidates, twins = exact_twins([_candidate(p) for p in photos if not p.reject])
    for dropped, kept in twins.items():
        by_path[dropped].judge_reason = f"= identical copy of {Path(kept).name}"
    distinct = {c.id for c in candidates}
    finalist_budget = 2 * args.roll_size
    if args.judge_coverage == "preselect":
        groups = {key: [_candidate(p) for p in group if str(p.path) in distinct]
                  for key, group in buckets.items()}
        reasons = {}
        keep = set(award_shortlist(groups, args.award_candidates, args.preselect_explore, reasons))
        for photo_id, reason in reasons.items():
            by_path[photo_id].preselection_reason = f"Award shortlist: {reason}"
        candidates = [c for c in candidates if c.id in keep]
    else:
        for c in candidates:
            by_path[c.id].preselection_reason = "Award roll: full coverage"
    # Pack the entire shortlist together: small day/species groups must not
    # each create a separate API call. Burst-aware packing remains available.
    rounds, _ = chunk_for_tournament(candidates, args.judge_chunk)
    requests, presentations = award_work_bound(rounds, finalist_budget, args.judge_chunk)
    print(f"Award roll: {sum(map(len, rounds))}/{len(distinct)} distinct eligible photos in the judge plan; "
          f"target {args.roll_size} across all dates")
    print(f"Judge plan: {len(rounds)} initial batches; at most {requests} successful requests and "
          f"{presentations} photo presentations including repeat comparisons. "
          "Detail crops and retries add work; these bounds are not a money limit.")
    if args.award_plan:
        for c in candidates:
            by_path[c.id].decision = "Planned for award judge (not sent)"
        for ph in photos:
            if ph.judge_reason:
                ph.decision = ph.judge_reason
        return dict(sorted(buckets.items()))
    judge.award_final = False
    finalists = _tournament(rounds, by_path, judge, "wildlife portfolio", "all dates", finalist_budget, args)
    judge.award_final = True
    try:
        winners = _judge_round(finalists, by_path, judge, "wildlife portfolio", "all dates",
                               args.roll_size, args, final=True)
    finally:
        judge.award_final = False
    for rank, photo_id in enumerate(winners, 1):
        by_path[photo_id].selected = True
        by_path[photo_id].award_rank = rank
    for ph in photos:
        if ph.selected:
            ph.decision = f"Award roll #{ph.award_rank}: {ph.judge_reason}"
        elif not ph.reject and ph.judge_reason:
            ph.decision = ph.judge_reason
    return dict(sorted(buckets.items()))


def select_photos(photos: list[Photo], args, judge=None) -> dict[tuple[str, str], list[Photo]]:
    if args.mode == "award-roll":
        return select_award_roll(photos, args, judge)
    buckets: dict[tuple[str, str], list[Photo]] = defaultdict(list)
    for ph in photos:
        ph.selected = ph.shortlisted = False
        ph.judge_reason = ph.preselection_reason = ""
        buckets[(ph.day, ph.label)].append(ph)
    by_path = {str(p.path): p for p in photos}
    if judge is not None:
        judge.sources = {str(p.path): p.source or p.path for p in photos}
    for day in sorted({d for d, _ in buckets}):
        day_buckets = {label: group for (d, label), group in buckets.items() if d == day}
        budgets = {label: pick_budget(sum(1 for p in g if not p.reject), args.max_per_group, args.extra_per)
                   for label, g in day_buckets.items()}
        cands = [_candidate(p) for g in day_buckets.values() for p in g if not p.reject]
        if judge is None:
            chosen = set(select_day(cands, budgets, args.dup_sim, args.dup_pixel, args.min_score,
                                    args.dup_reframe_sim))
        else:
            chosen = set()
            groups = tqdm(budgets.items(), desc=f"Judging {day}", unit="group", leave=False)
            for label, k in groups:
                groups.set_postfix_str(label, refresh=False)
                sc = [_candidate(p) for p in day_buckets[label] if not p.reject]
                # Record all exact twins even when preselection is requested.
                sc, twins = exact_twins(sc)
                n_distinct = len(sc)
                for dropped, kept in twins.items():
                    by_path[dropped].judge_reason = f"= identical copy of {Path(kept).name}"
                if args.judge_coverage == "preselect":
                    cap = max(args.preselect_min, int(round(args.preselect * k)),
                              math.ceil(args.preselect_share * len(sc)))
                    reasons = {}
                    keep = set(diverse_preselection(sc, cap, explore_slots=args.preselect_explore, reasons=reasons))
                    for photo_id, reason in reasons.items():
                        by_path[photo_id].preselection_reason = reason
                    for c in sc:
                        if c.id not in keep:
                            by_path[c.id].judge_reason = "Outside judge shortlist"
                    sc = [c for c in sc if c.id in keep]
                    if sc:
                        tqdm.write(f"Judge shortlist: {day} {label}: {len(sc)}/{n_distinct} distinct frames")
                rounds, _ = chunk_for_tournament(sc, args.judge_chunk)
                chosen.update(_tournament(rounds, by_path, judge, label, day, k, args))
            groups.set_postfix_str("cross-label duplicates", refresh=True)
            chosen = _reconcile_day(chosen, by_path, judge, day, args)
            groups.close()
        for group in day_buckets.values():
            for p in group:
                p.selected = str(p.path) in chosen
                if p.reject:
                    p.decision = f"Rejected: {p.reject}"
                elif judge is not None:
                    p.decision = "Selected by judge" if p.selected else p.judge_reason
                elif p.selected:
                    p.decision = "Selected by score"
                elif p.score < args.min_score:
                    p.decision = "Below requested score cutoff"
                else:
                    twin = next((i for i in sorted(chosen)
                                 if is_duplicate(_candidate(p), [_candidate(by_path[i])],
                                                 args.dup_sim, args.dup_pixel, args.dup_reframe_sim)), None)
                    p.decision = f"Similar to selected {Path(twin).name}" if twin else "Group budget filled"
    return dict(sorted(buckets.items()))


def write_report(photos: list[Photo], path: Path, root: Path | None = None) -> None:
    root = root or path.parent
    cols = ["file", "datetime", "day", "label", "label_prob", "person", "subject", "edge", "scene", "sharpness",
            "clip_low", "clip_high", "aesthetic", "score", "reject", "shortlisted", "selected", "judge_reason",
            "iso", "edit_score", "edit_benefit", "edit_why", "preset", "relative_file", "source",
            "subject_sharpness", "decision", *GEOMETRY_FIELDS, "placement_reliability", "composition_score",
            "composition_bonus", *[f"editorial_{name}" for name in EDITORIAL_PROMPTS],
            "editorial_score", "editorial_bonus", "moment", "moment_novelty", "subject_embedding",
            "focus_rank", "detail_focus", "refined_focus_adjustment", "focus_refinement", "preselection_reason",
            "burst", "subject_box_count", "interaction_embedding", *[f"burst_{name}" for name in BURST_PROMPTS],
            "judge_label", "award_rank", "award_moment", "award_special_moment", "award_exceptional_reason",
            *REVIEW_FIELDS]
    def metric(value):
        return f"{value:.4f}" if value is not None else ""
    with io.StringIO(newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for ph in sorted(photos, key=lambda p: (p.day, p.label, -p.score)):
            relative = os.path.relpath(ph.path, root)
            digest = ph.digest if not ph.source or ph.source == ph.path else ""
            try:
                if not digest and ph.path.is_file():
                    digest = fingerprint(ph.path)
                identity = photo_id(relative, digest) if digest else ""
            except (OSError, ReviewError):
                # The report is still written; this row simply cannot carry a review identity.
                identity = digest = ""
            wr.writerow([
                str(ph.path), ph.time.isoformat() if ph.time else "", ph.day, ph.label,
                f"{ph.label_prob:.2f}", f"{ph.person:.3f}", f"{ph.subject:.3f}", int(ph.edge), ph.scene,
                f"{ph.sharpness:.1f}", f"{ph.clip_low:.3f}", f"{ph.clip_high:.3f}", f"{ph.aesthetic:.2f}",
                f"{ph.score:.3f}", ph.reject, int(ph.shortlisted), int(ph.selected), ph.judge_reason,
                ph.iso or "", ph.edit_score, ph.edit_band, ph.edit_why, ph.preset,
                os.path.relpath(ph.path, root), str(ph.source or ph.path),
                f"{ph.subject_sharpness:.2f}" if ph.subject_sharpness is not None else "",
                ph.decision or ph.reject,
                *[metric(ph.composition.get(name)) for name in GEOMETRY_FIELDS],
                metric(ph.composition.get("placement_reliability")), metric(ph.composition.get("composition_score")),
                metric(ph.composition_bonus), *[metric(ph.editorial.get(name)) for name in EDITORIAL_PROMPTS],
                metric(ph.editorial.get("editorial_score")), metric(ph.editorial_bonus),
                ph.moment, metric(ph.moment_novelty), int(ph.subject_dino is not None), metric(ph.focus_rank),
                metric(ph.detail_focus), metric(ph.refined_focus_adjustment),
                ph.focus_refinement, ph.preselection_reason,
                ph.burst, len(ph.subject_boxes), int(ph.interaction_dino is not None),
                *[metric(ph.burst_signals.get(name)) for name in BURST_PROMPTS],
                ph.judge_label,
                ph.award_rank or "",
                ph.award_assessment.moment if ph.award_assessment else "",
                int(ph.award_assessment.special_moment) if ph.award_assessment else "",
                ph.award_assessment.exceptional_reason if ph.award_assessment else "",
                identity, digest,
                int(ph.selected if ph.auto_selected is None else ph.auto_selected),
                (ph.auto_decision if ph.auto_selected is not None else ph.decision) or ph.reject,
                ph.manual_decision, ph.manual_reason, ph.review_status, ph.review_revision,
            ])
        atomic_write(path, f.getvalue())


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
        if ph.selected and ph.preset:
            keywords.append(f"FotoSort|Preset|{ph.preset}")
        if ph.selected and ph.edit_band:
            keywords.append(f"FotoSort|RAW edit {ph.edit_band}")
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
        ph.time, ph.iso = read_exif(ph.path)

    if args.apply_report:
        return apply_report(photos, root, args)

    # Read manual decisions before any expensive analysis or judge call can be wasted on a bad store.
    try:
        review_state = ReviewStore(root).load()
    except (OSError, ReviewError) as exc:
        print(f"Review error: {exc}", file=sys.stderr)
        return 2

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
    assign_moments(photos, args)
    score_and_reject(photos, args)
    forced = forced_includes(photos, args.include)
    if args.taste_dir:
        taste_bonus(photos, args.taste_dir, args.taste_weight, emb)
    refine_close_focus(photos, root, args)
    judge = None
    if args.judge and not args.award_plan:
        from fotosort.judge import Judge

        judge = Judge(args.judge_provider, args.judge_model, args.key_file, args.judge_detail, args.judge_hint,
                      args.judge_base_url, args.judge_crops, args.mode)
        print(f"Judging {args.judge_coverage} coverage with {judge.model} ...")
    buckets = select_photos(photos, args, judge)
    if args.award_plan:
        report = root / args.report
        write_report(photos + unreadable, report, root)
        print(f"Local candidate plan: {report}. No judge calls or photo exports.")
        return 0
    for ph in forced:
        # --include is a human choice: keep the automatic result separate, as for manual decisions.
        ph.auto_selected, ph.auto_decision = ph.selected, ph.decision or ph.reject
        ph.selected, ph.judge_reason = True, "forced by --include"
        ph.decision = "Forced by --include"
    for warning in apply_manual_decisions(photos, root, review_state):
        print(f"Review: {warning}", file=sys.stderr)
    if judge is not None and args.judge_species and args.mode != "award-roll":
        name_pick_subjects([ph for ph in photos if ph.selected], judge, labels)
    if judge is not None:
        print(f"Judge tokens: {judge.usage['input']} in, {judge.usage['output']} out")

    editing_advice(photos, emb, args)
    report = root / args.report
    write_report(photos + unreadable, report, root)
    if args.xmp:
        write_sidecars(photos, args)
    if args.raw_cull:
        raw_cull(photos, root, args)

    picks = [ph for ph in photos if ph.selected]
    if args.mode == "award-roll":
        picks.sort(key=lambda ph: ph.award_rank)
        print(f"Award winning camera roll: {len(picks)}/{args.roll_size} photos")
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


def name_pick_subjects(picks: list[Photo], judge, labels: list[str]) -> dict[str, str]:
    """The judge names each pick's subject at full detail; CLIP's guess stays as
    fallback. Returns {path: judge label} for the picks it could name."""
    if not picks or not hasattr(judge, "name_subjects"):
        return {}
    named = judge.name_subjects([ph.path for ph in tqdm(picks, desc="Naming subjects", unit="img", leave=False)],
                                labels)
    changed = 0
    for ph in picks:
        ph.judge_label = named.get(str(ph.path), "")
        changed += bool(ph.judge_label) and bucket_name(ph.judge_label) != ph.label
    print(f"Judge named {len(named)} of {len(picks)} picks; {changed} differ from the local label")
    return named


def layout_dir(out_dir: Path, ph: Photo, layout: str) -> Path:
    """Where a pick goes: flat, per subject, per day, or day/subject. The judge's
    subject name wins over the local CLIP label when it exists."""
    label = bucket_name(ph.judge_label) if ph.judge_label else ph.label
    safe = "".join(c if c.isalnum() or c in " -_()" else "_" for c in label).strip() or "other"
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

    group = [ph for ph in (only if only is not None else photos) if raw_sibling(ph.path) is not None]
    raws = {id(ph): raw_sibling(ph.path) for ph in group}
    if not group:
        return
    missing = []
    for ph in group:
        twin = find_twin(raws[id(ph)], args.dxo_dir)
        if twin is not None:
            ph.source = twin
        else:
            missing.append(ph)
    print(f"DxO PureRAW: {len(group) - len(missing)} of {len(group)} RAW files already processed")
    if missing and args.dxo_wait > 0:
        print(f"Handing {len(missing)} RAW files to {args.dxo_app}: press Process there. "
              f"Waiting up to {args.dxo_wait:g} h for the outputs ...")
        try:
            send_to_pureraw(list(dict.fromkeys(raws[id(ph)] for ph in missing)), args.dxo_app)
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"Could not open {args.dxo_app}: {e}; continuing with the originals")
        else:
            done = wait_for_twins(list(dict.fromkeys(raws[id(ph)] for ph in missing)),
                                  args.dxo_dir, args.dxo_wait * 3600)
            for ph in missing:
                if raws[id(ph)] in done:
                    ph.source = done[raws[id(ph)]]
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
    by_relative = {r["relative_file"]: r for r in rows.values() if r.get("relative_file")}
    # Legacy reports contain only absolute paths. Multiple subfolders let us
    # reconstruct their common old root without collapsing camera filenames.
    absolute = [Path(k) for k in rows if Path(k).is_absolute()]
    old_root = Path(os.path.commonpath([str(p.parent) for p in absolute])) if absolute else None
    legacy_relative = ({str(Path(k).relative_to(old_root)): r for k, r in rows.items()
                        if Path(k).is_absolute() and not r.get("relative_file")} if old_root else {})
    by_name, current_names = defaultdict(list), defaultdict(list)
    for name, row in rows.items():
        if not row.get("relative_file"):
            by_name[Path(name).name].append(row)
    for ph in photos:
        current_names[ph.path.name].append(ph)
    matches = []
    for ph in photos:
        relative = str(ph.path.relative_to(root))
        r = rows.get(str(ph.path)) or by_relative.get(relative) or legacy_relative.get(relative)
        if r is None and ph.path.name in by_name:
            if len(by_name[ph.path.name]) != 1 or len(current_names[ph.path.name]) != 1:
                raise SystemExit(f"Ambiguous report filename {ph.path.name}; use relative_file paths "
                                 "to identify the correct photos before applying this report.")
            r = by_name[ph.path.name][0]
        matches.append((ph, r))
    # Resolve every row before allowing copy, move, enhancement or RAW culling.
    # Only picks are copied, moved or enhanced, so only their contents need to match the report.
    for ph, row in matches:
        if not row or row.get("selected") != "1" or not row.get("photo_sha256"):
            continue
        try:
            changed = fingerprint(ph.path) != row["photo_sha256"]
        except (OSError, ReviewError):
            changed = True
        if changed:
            raise SystemExit(f"Photo changed since this report: {ph.path.name}; rerun analysis before applying it.")
    picks, matched = [], set()
    for ph, r in matches:
        if r is not None:
            matched.add(id(r))
            ph.label = r.get("label", "")
            ph.judge_label = r.get("judge_label", "")
            ph.edit_band, ph.preset = r.get("edit_benefit", ""), r.get("preset", "")
        if r and r.get("selected") == "1":
            ph.person = float(r.get("person") or 0)
            ph.selected = True
            picks.append(ph)
    missing = sum(r.get("selected") == "1" and id(r) not in matched for r in rows.values())
    print(f"{len(picks)} picks from the report" + (f" ({missing} listed files no longer exist)" if missing else ""))
    if any(ph.selected and not ph.edit_band for ph in photos):
        # report from an older run: fill the heuristic benefit (and a style preset) from the cache
        cached = cache_mod.load(root)
        for ph in photos:
            c = cached.get(ph.key)
            if c is None:
                continue
            ph.emb, ph.clip_low, ph.clip_high, ph.mean = c["emb"], c["clip_low"], c["clip_high"], c["mean"]
            ph.person = c["person"]
        emb_for_style = None
        if any(ph.selected and not ph.preset for ph in photos):
            from fotosort.embed import Embedder

            emb_for_style = Embedder()
        editing_advice(photos, emb_for_style, args)
    if args.judge and args.judge_species and any(not ph.judge_label for ph in picks):
        from fotosort.judge import Judge

        judge = Judge(args.judge_provider, args.judge_model, args.key_file, args.judge_detail, args.judge_hint,
                      args.judge_base_url, args.judge_crops)
        named = name_pick_subjects([ph for ph in picks if not ph.judge_label], judge, load_labels(args.labels))
        print(f"Judge tokens: {judge.usage['input']} in, {judge.usage['output']} out")
        if named:
            update_report_column(report, "judge_label", named)
    if args.raw_cull:
        raw_cull(photos, root, args)
    if not (args.move or args.copy):
        for ph in picks:
            print(f"  {ph.path.relative_to(root)}  [{ph.judge_label or ph.label}]")
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


def update_report_column(report: Path, column: str, values: dict[str, str]) -> None:
    """Write `values` ({file column value: new value}) into one column of the report,
    adding the column to reports from older runs. Other cells stay untouched."""
    with open(report, newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = list(reader.fieldnames or []), list(reader)
    if column not in fields:
        fields.append(column)
    for r in rows:
        r.setdefault(column, "")
        if r["file"] in values:
            r[column] = values[r["file"]]
    tmp = report.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    os.replace(tmp, report)


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

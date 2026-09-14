"""Per-folder cache of per-image features so re-runs skip the model."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from fotosort.burst import MAX_SUBJECT_BOXES, valid_boxes

CACHE_NAME = ".fotosort_cache.npz"
CACHE_VERSION = 8


def cache_key(path: Path, root: Path) -> str:
    st = path.stat()
    return f"{os.path.relpath(path, root)}|{st.st_size}|{st.st_mtime_ns}"


def load(root: Path, detector: str = "") -> dict[str, dict]:
    """Entries of the folder cache, or {} if it was written by another version
    or with a different subject detector (its person/subject fields would be stale)."""
    p = root / CACHE_NAME
    if not p.exists():
        return {}
    try:
        with np.load(p, allow_pickle=False) as archive:
            if "version" not in archive or int(archive["version"]) not in (6, 7, CACHE_VERSION):
                return {}
            if detector and str(archive["detector"]) != detector:
                print(f"Cache was built with detector {archive['detector']!s}, now {detector}: recomputing")
                return {}
            # NPZ lookup reads a whole array each time. Materialise each once;
            # rows then share these allocations instead of retaining N copies.
            z = {name: archive[name] for name in archive.files}
        keys = z["keys"]
        return {
            str(k): {
                "emb": z["emb"][i],
                "sig": z["sig"][i],
                "dino": z["dino"][i],
                "subject_dino": (z["subject_dino"][i] if "subject_dino" in z
                                 and np.isfinite(z["subject_dino"][i]).all() else None),
                "subject_dino_checked": bool(z["subject_dino_checked"][i])
                if "subject_dino_checked" in z else False,
                **{name: (z[name][i] if name in z and np.isfinite(z[name][i]).all() else None)
                   for name in ("subject_clip", "interaction_clip", "interaction_dino")},
                "subject_boxes": valid_boxes(z["subject_boxes"][i]) if "subject_boxes" in z else [],
                "subject_boxes_checked": bool(z["subject_boxes_checked"][i])
                if "subject_boxes_checked" in z else False,
                "burst_features_checked": bool(z["burst_features_checked"][i])
                if "burst_features_checked" in z else False,
                "detail_focus": (float(z["detail_focus"][i]) if "detail_focus" in z
                                 and np.isfinite(z["detail_focus"][i]) else None),
                "detail_focus_checked": bool(z["detail_focus_checked"][i])
                if "detail_focus_checked" in z else False,
                "person": float(z["person"][i]),
                "subject": float(z["subject"][i]),
                "edge": bool(z["edge"][i]),
                "sharpness": float(z["sharpness"][i]),
                "clip_low": float(z["clip_low"][i]),
                "clip_high": float(z["clip_high"][i]),
                "mean": float(z["mean"][i]),
                "digest": str(z["digest"][i]),
                "box": tuple(z["box"][i]) if np.isfinite(z["box"][i]).all() else None,
                "subject_sharpness": (float(z["subject_sharpness"][i])
                                      if np.isfinite(z["subject_sharpness"][i]) else None),
            }
            for i, k in enumerate(keys)
        }
    except Exception:
        return {}


def save(root: Path, entries: dict[str, dict], detector: str = "") -> None:
    if not entries:
        return
    keys = list(entries)
    boxes = np.full((len(keys), MAX_SUBJECT_BOXES, 4), np.nan, dtype=np.float32)
    for i, key in enumerate(keys):
        detected = valid_boxes(entries[key].get("subject_boxes", []))
        if detected:
            boxes[i, :len(detected)] = detected
    arrays = dict(
        version=np.array(CACHE_VERSION),
        detector=np.array(detector),
        keys=np.array(keys),
        sig=np.stack([entries[k]["sig"] for k in keys]).astype(np.float32),
        dino=np.stack([entries[k]["dino"] for k in keys]).astype(np.float32),
        subject_dino=np.stack([entries[k]["subject_dino"] if entries[k].get("subject_dino") is not None
                               else np.full_like(entries[k]["dino"], np.nan, dtype=np.float32)
                               for k in keys]).astype(np.float32),
        subject_dino_checked=np.array([entries[k].get("subject_dino_checked", False) for k in keys]),
        subject_boxes=boxes,
        subject_boxes_checked=np.array([entries[k].get("subject_boxes_checked", False) for k in keys]),
        burst_features_checked=np.array([entries[k].get("burst_features_checked", False) for k in keys]),
        **{name: np.stack([entries[k][name] if entries[k].get(name) is not None
                          else np.full_like(entries[k][like], np.nan, dtype=np.float32) for k in keys])
           for name, like in (("subject_clip", "emb"), ("interaction_clip", "emb"), ("interaction_dino", "dino"))},
        detail_focus=np.array([entries[k].get("detail_focus") for k in keys], dtype=np.float32),
        detail_focus_checked=np.array([entries[k].get("detail_focus_checked", False) for k in keys]),
        person=np.array([entries[k]["person"] for k in keys], dtype=np.float32),
        subject=np.array([entries[k]["subject"] for k in keys], dtype=np.float32),
        edge=np.array([entries[k]["edge"] for k in keys], dtype=bool),
        emb=np.stack([entries[k]["emb"] for k in keys]).astype(np.float32),
        sharpness=np.array([entries[k]["sharpness"] for k in keys], dtype=np.float32),
        clip_low=np.array([entries[k]["clip_low"] for k in keys], dtype=np.float32),
        clip_high=np.array([entries[k]["clip_high"] for k in keys], dtype=np.float32),
        mean=np.array([entries[k]["mean"] for k in keys], dtype=np.float32),
        digest=np.array([entries[k].get("digest", "") for k in keys]),
        box=np.array([entries[k].get("box") or (np.nan,) * 4 for k in keys], dtype=np.float32),
        subject_sharpness=np.array([entries[k].get("subject_sharpness") for k in keys], dtype=np.float32),
    )
    # An interrupted save must not destroy the last usable cache.
    with tempfile.NamedTemporaryFile(dir=root, prefix=".fotosort-cache-", suffix=".npz", delete=False) as tmp:
        name = Path(tmp.name)
    try:
        np.savez(name, **arrays)
        os.replace(name, root / CACHE_NAME)
    finally:
        name.unlink(missing_ok=True)

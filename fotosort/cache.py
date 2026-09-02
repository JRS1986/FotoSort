"""Per-folder cache of per-image features so re-runs skip the model."""
from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE_NAME = ".fotosort_cache.npz"


def cache_key(path: Path, root: Path) -> str:
    st = path.stat()
    return f"{path.relative_to(root)}|{st.st_size}|{int(st.st_mtime)}"


def load(root: Path) -> dict[str, dict]:
    p = root / CACHE_NAME
    if not p.exists():
        return {}
    try:
        z = np.load(p, allow_pickle=False)
        keys = z["keys"]
        return {
            str(k): {
                "emb": z["emb"][i],
                "sharpness": float(z["sharpness"][i]),
                "clip_low": float(z["clip_low"][i]),
                "clip_high": float(z["clip_high"][i]),
                "mean": float(z["mean"][i]),
            }
            for i, k in enumerate(keys)
        }
    except Exception:
        return {}


def save(root: Path, entries: dict[str, dict]) -> None:
    if not entries:
        return
    keys = list(entries)
    np.savez(
        root / CACHE_NAME,
        keys=np.array(keys),
        emb=np.stack([entries[k]["emb"] for k in keys]).astype(np.float32),
        sharpness=np.array([entries[k]["sharpness"] for k in keys], dtype=np.float32),
        clip_low=np.array([entries[k]["clip_low"] for k in keys], dtype=np.float32),
        clip_high=np.array([entries[k]["clip_high"] for k in keys], dtype=np.float32),
        mean=np.array([entries[k]["mean"] for k in keys], dtype=np.float32),
    )

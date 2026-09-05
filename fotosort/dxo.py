"""DxO PureRAW as a pre-processing step.

PureRAW has no command line. What it does offer: files handed to it with
`open -a` land in its batch list, and after you press Process it writes one
output per RAW into a `DxO` folder next to the originals (or wherever its
output setting points), named with your template, e.g. `20260831-_8311254.dng`
or `_8311254-DxO_DeepPRIME.dng`.

So the integration is: find the processed twin of a RAW if it exists; if not,
hand the RAW to PureRAW and wait for the twin to appear. Everything downstream
(scoring, enhancement, copying) then uses the twin, while the original RAW
keeps its identity in the report and sidecars.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

OUTPUT_EXT = (".dng", ".tif", ".tiff", ".jpg", ".jpeg")


def dxo_dir_for(raw: Path, dxo_dir: Path | None) -> Path:
    return dxo_dir if dxo_dir is not None else raw.parent / "DxO"


def find_twin(raw: Path, dxo_dir: Path | None = None) -> Path | None:
    """The PureRAW output for `raw`: a file in the DxO folder whose stem is the
    RAW's stem, or ends with `-<stem>` (date-prefixed template), or starts with
    `<stem>-DxO` (suffix template). DNG is preferred over TIFF over JPEG."""
    folder = dxo_dir_for(raw, dxo_dir)
    if not folder.is_dir():
        return None
    stem = raw.stem
    candidates = []
    for f in folder.iterdir():
        if f.suffix.lower() not in OUTPUT_EXT or f.name.startswith("."):
            continue
        s = f.stem
        if s == stem or s.endswith("-" + stem) or s.endswith("_" + stem) or s.startswith(stem + "-DxO") \
                or s.startswith(stem + "_DxO"):
            candidates.append(f)
    if not candidates:
        return None
    return sorted(candidates, key=lambda f: OUTPUT_EXT.index(f.suffix.lower()))[0]


def send_to_pureraw(files: list[Path], app: str = "PureRAW 6") -> None:
    """Add files to PureRAW's batch list. The user still has to press Process."""
    if not files:
        return
    subprocess.run(["open", "-a", app, *map(str, files)], check=True)


def _stable(path: Path, seconds: float = 5.0) -> bool:
    """True when the file exists and its size did not change for `seconds`."""
    try:
        s1 = path.stat().st_size
    except OSError:
        return False
    time.sleep(seconds)
    try:
        return path.stat().st_size == s1 and s1 > 0
    except OSError:
        return False


def wait_for_twins(raws: list[Path], dxo_dir: Path | None, timeout_s: float, poll_s: float = 15.0,
                   log=print) -> dict[Path, Path]:
    """Poll until every RAW has a finished twin or the timeout passes.
    Returns {raw: twin} for the ones that appeared."""
    pending = list(raws)
    done: dict[Path, Path] = {}
    start = time.time()
    last_report = 0
    while pending and time.time() - start < timeout_s:
        still = []
        for raw in pending:
            twin = find_twin(raw, dxo_dir)
            if twin is not None and _stable(twin, 2.0):
                done[raw] = twin
            else:
                still.append(raw)
        pending = still
        if pending and time.time() - last_report > 60:
            log(f"Waiting for PureRAW: {len(done)}/{len(raws)} done, {len(pending)} to go ...")
            last_report = time.time()
        if pending:
            time.sleep(poll_s)
    return done

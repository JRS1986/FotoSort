"""Local RightWayUp integration, loaded only for enhanced photo exports."""
from __future__ import annotations

import argparse
import math

from PIL import Image

MIN_ANGLE = 1.0  # degrees; the default of upstream `rightwayup fix --min-angle`


def add_orientation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-orientation", action="store_true",
                        help="Disable RightWayUp orientation correction during enhancement")
    parser.add_argument("--orientation-tier", default="max",
                        choices=("pico", "nano", "fast", "balanced", "pro", "max"),
                        help="RightWayUp model tier (default: max; weights download on first use)")
    parser.add_argument("--orientation-snap", type=int, choices=(0, 90), default=0,
                        help="0 = any angle with empty corners cropped (default); 90 = quarter turns")
    parser.add_argument("--orientation-min-angle", type=_min_angle, default=MIN_ANGLE,
                        help="Treat residual tilts below this many degrees as level, like 'rightwayup fix' "
                             f"(default: {MIN_ANGLE}; 0 applies every estimate)")


def _min_angle(value: str) -> float:
    angle = float(value)
    if not 0 <= angle < 45:
        raise argparse.ArgumentTypeError("must be at least 0 and below 45 degrees")
    return angle


def load_orienter(args, strength: float):
    """Initialize once, before export operations; disabled paths never import the model."""
    if args.no_orientation or strength <= 0:
        return None
    try:
        from rightwayup import Orienter

        print(f"Loading RightWayUp ({args.orientation_tier}, strict abstention) ...")
        return Orienter(tier=args.orientation_tier, abstain="strict")
    except ImportError as exc:
        raise SystemExit("RightWayUp is required for orientation correction. Reinstall FotoSort or run "
                         "'pip install \"rightwayup>=1.0,<2\"'; --no-orientation disables it.") from exc
    except Exception as exc:
        raise SystemExit("Could not load RightWayUp. Check the model download/cache or RIGHTWAYUP_MODEL_DIR; "
                         f"--no-orientation disables it. {exc}") from exc


def orientation_details(result, orienter, snap: int, min_angle: float = MIN_ANGLE) -> dict:
    """Describe the exact correction applied with RightWayUp's correct() API.

    A residual tilt below `min_angle` is within the model's error: it becomes
    the nearest quarter turn rather than rotating and cropping a level photo.
    """
    angle, confidence = float(result.angle_cw), float(result.confidence)
    if not (math.isfinite(angle) and math.isfinite(confidence)):
        raise ValueError("RightWayUp returned a non-finite angle or confidence")
    # Upstream decoding can round to exactly 360.0, and float32 sums can slightly exceed 1.
    angle, confidence = angle % 360, min(max(confidence, 0.0), 1.0)
    quarter = round(angle / 90) * 90
    correction = quarter % 360 if snap or abs(angle - quarter) < min_angle else angle
    if result.abstain:
        correction = 0
    return {
        "orientation_status": "abstained" if result.abstain else ("corrected" if correction else "upright"),
        "orientation_angle_cw": angle,
        "orientation_confidence": confidence,
        "orientation_abstain": int(result.abstain),
        "orientation_correction_ccw": correction,
        "orientation_tier": result.tier,
        "orientation_routed": int(result.routed),
        "orientation_precision": orienter.precision,
        "orientation_device": orienter.device,
        "orientation_snap": snap,
        "orientation_min_angle": min_angle,
    }


def crop_rotation_corners(rotated: Image.Image, original_size: tuple[int, int], angle: float) -> Image.Image:
    """Centered inscribed crop, retaining the aspect ratio after the nearest quarter turn.

    A crop's corners must fit inside the source in both axes when rotated back.
    Reserve two source pixels on each edge for bicubic interpolation. Quarter
    turns need neither interpolation nor cropping.
    """
    quarter_turns = round(angle / 90)
    tilt = abs(angle - quarter_turns * 90)
    if tilt == 0:
        return rotated
    width, height = original_size
    if quarter_turns % 2:
        width, height = height, width
    cosine, sine = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
    scale = min(max(1, width - 4) / (width * cosine + height * sine),
                max(1, height - 4) / (width * sine + height * cosine))
    crop_width, crop_height = max(1, math.floor(width * scale)), max(1, math.floor(height * scale))
    left, top = (rotated.width - crop_width) // 2, (rotated.height - crop_height) // 2
    return rotated.crop((left, top, left + crop_width, top + crop_height))

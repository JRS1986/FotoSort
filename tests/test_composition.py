"""Composition preferences must help nomination without becoming hard rules."""
import csv
from pathlib import Path

import numpy as np
import pytest

from fotosort.cli import Photo, _candidate, parse_args, score_and_reject, select_photos, write_report
from fotosort.composition import EDITORIAL_PROMPTS, GOLDEN_MINOR, composition_metrics, editorial_scores
from fotosort.selection import ScenedCandidate, diverse_preselection


def box_at(x, y, half=0.04):
    return x - half, y - half, x + half, y + half


@pytest.mark.parametrize("x,y", [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)])
def test_thirds_intersections_have_full_affinity(x, y):
    assert composition_metrics(box_at(x, y))["rule_of_thirds"] == pytest.approx(1)


@pytest.mark.parametrize("x,y", [(GOLDEN_MINOR, GOLDEN_MINOR), (1 - GOLDEN_MINOR, GOLDEN_MINOR),
                                  (GOLDEN_MINOR, 1 - GOLDEN_MINOR), (1 - GOLDEN_MINOR, 1 - GOLDEN_MINOR)])
def test_golden_ratio_intersections_have_full_affinity(x, y):
    assert composition_metrics(box_at(x, y))["golden_ratio"] == pytest.approx(1)


def test_centered_composition_is_an_equal_alternative_and_bonuses_do_not_stack():
    center = composition_metrics(box_at(0.5, 0.5))
    thirds = composition_metrics(box_at(1 / 3, 1 / 3))
    golden = composition_metrics(box_at(GOLDEN_MINOR, GOLDEN_MINOR))
    assert center["center_composition"] == pytest.approx(1)
    assert center["composition_score"] == thirds["composition_score"] == golden["composition_score"]
    assert 0 < golden["rule_of_thirds"] < 1  # overlaps the broad thirds guide
    assert golden["composition_score"] == pytest.approx(1)  # no double-counting


def test_geometry_is_mirror_invariant_and_tapers_smoothly():
    a = composition_metrics(box_at(0.3, 0.7))
    b = composition_metrics(box_at(0.7, 0.3))
    assert a["rule_of_thirds"] == pytest.approx(b["rule_of_thirds"])
    assert a["golden_ratio"] == pytest.approx(b["golden_ratio"])
    peak = composition_metrics(box_at(1 / 3, 1 / 3))["rule_of_thirds"]
    near = composition_metrics(box_at(1 / 3 + 0.01, 1 / 3 + 0.01))["rule_of_thirds"]
    far = composition_metrics(box_at(0.1, 0.1))["rule_of_thirds"]
    assert peak > near > far and near > 0.9


@pytest.mark.parametrize("box", [None, (0, 0, 0, 0), (-0.1, 0, 0.5, 0.5), (0, 0, 1.1, 1),
                                 (0.8, 0.2, 0.1, 0.9), (0, 0, float("nan"), 1), (0, 0, 1)])
def test_invalid_or_missing_detections_are_unknown_not_bad_compositions(box):
    assert all(value is None for value in composition_metrics(box).values())


def test_frame_filling_box_does_not_pretend_to_locate_a_focal_point():
    metrics = composition_metrics((0, 0, 1, 1))
    assert metrics["placement_reliability"] == 0
    assert metrics["composition_score"] == 0


def test_editorial_contrasts_are_normalized_embedding_margins_not_probabilities():
    dim = len(EDITORIAL_PROMPTS)
    texts = np.stack([sign * v for v in np.eye(dim) for sign in (1, -1)])
    scores = editorial_scores(np.array([[1., 0, 0, 0], [-1., 0, 0, 0]]), texts)
    assert scores["light"].tolist() == [2, -2]
    assert scores["editorial_score"].tolist() == [0.5, -0.5]
    with pytest.raises(ValueError):
        editorial_scores(np.ones((1, dim)), texts[:2])


def photo(name, box=None, editorial=0):
    return Photo(Path(name), name, label="bird", aesthetic=5, sharpness=100, box=box,
                 emb=np.array([1., 0.]), dino=np.array([1., 0.]), editorial={"editorial_score": editorial})


def test_composition_changes_local_choice_and_respects_disabled_weight(tmp_path):
    # Equal aesthetics and focus: prefer the deliberately placed subject.
    frames = [photo("edge.jpg", box_at(0.08, 0.08)), photo("thirds.jpg", box_at(1 / 3, 1 / 3))]
    args = parse_args([str(tmp_path), "--max-per-group", "1"])
    score_and_reject(frames, args)
    select_photos(frames, args)
    assert frames[1].selected and not frames[0].selected
    assert 0 <= frames[0].composition_bonus < frames[1].composition_bonus <= args.composition_weight
    args.composition_weight = 0
    score_and_reject(frames, args)
    assert frames[0].score == frames[1].score
    assert all(_candidate(p).composition == 0 for p in frames)


def test_editorial_bonus_is_bounded_and_disabled_weight_disables_nominations(tmp_path):
    frames = [photo(f"{i}.jpg", editorial=i * 0.01) for i in range(10)]
    args = parse_args([str(tmp_path)])
    score_and_reject(frames, args)
    assert frames[-1].editorial_bonus > frames[0].editorial_bonus
    assert all(0 <= p.editorial_bonus <= args.editorial_weight for p in frames)
    args.editorial_weight = 0
    score_and_reject(frames, args)
    assert all(p.editorial_bonus == 0 and _candidate(p).editorial == 0 for p in frames)


def test_nonconforming_composition_is_not_rejected_and_stronger_photo_still_wins(tmp_path):
    frames = [photo("exceptional.jpg", box_at(0.08, 0.08)), photo("ordinary.jpg", box_at(1 / 3, 1 / 3))]
    frames[0].aesthetic = 8
    args = parse_args([str(tmp_path)])
    score_and_reject(frames, args)
    assert frames[0].score > frames[1].score
    assert not any(p.reject for p in frames)
    missing = photo("no-detection.jpg")
    score_and_reject([missing], args)
    assert missing.composition_bonus == 0 and not missing.reject


def test_independent_nominations_rescue_aesthetic_candidates_within_same_shortlist_budget():
    frames = [ScenedCandidate(str(i), 30 - i, "bird", np.array([1., 0.]), scene=i % 2,
                              aesthetic=0, editorial=0, composition=0) for i in range(30)]
    frames[-1].aesthetic = 9
    frames[-2].editorial = 0.15
    frames[-3].composition = 0.2
    shortlisted = diverse_preselection(frames, cap=12)
    assert len(shortlisted) == len(set(shortlisted)) == 12
    assert {"0", "1", "29", "28", "27"} <= set(shortlisted)
    # Scene coverage takes precedence when the allocation is already full.
    for i, frame in enumerate(frames):
        frame.scene = i
    assert len(diverse_preselection(frames, cap=12)) == 30


def test_report_contains_geometry_editorial_margins_and_applied_bonuses(tmp_path):
    p = photo("test.jpg", box_at(GOLDEN_MINOR, GOLDEN_MINOR), 0.04)
    p.editorial.update(light=0.1, composition=0.02, subject=0.01, moment=0.03)
    args = parse_args([str(tmp_path)])
    score_and_reject([p], args)
    report = tmp_path / "report.csv"
    write_report([p], report, tmp_path)
    with report.open() as stream:
        row = next(csv.DictReader(stream))
    assert row["golden_ratio"] == "1.0000"
    assert row["composition_bonus"] == "0.2000"
    assert row["editorial_moment"] == "0.0300"
    assert row["editorial_score"] == "0.0400"


@pytest.mark.parametrize("flag", ["--composition-weight", "--editorial-weight"])
@pytest.mark.parametrize("value", ["-1", "2", "nan"])
def test_invalid_preference_weights_are_rejected(flag, value):
    with pytest.raises(SystemExit):
        parse_args(["/tmp", flag, value])

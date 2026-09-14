"""Keeper-recall regressions, using a deterministic judge rather than model calls."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from fotosort.cli import Photo, parse_args, score_and_reject, select_photos
from fotosort.judge import Verdict
from fotosort.quality import sharpness, signature, subject_focus, to_gray
from fotosort.selection import ScenedCandidate, chunk_for_tournament, diverse_preselection


def photos_in(tmp_path, n):
    out = []
    for i in range(n):
        path = tmp_path / f"{i:03}.jpg"
        path.touch()
        vec = np.eye(n, dtype=np.float32)[i]
        out.append(Photo(path, str(i), time=datetime(2026, 9, 1), label="bird", scene=i,
                         aesthetic=float(n - i), sharpness=100, score=n - i, emb=vec, dino=vec))
    return out


class Oracle:
    def __init__(self):
        self.calls = []

    def judge(self, paths, label, day, k, boxes, final=False):
        self.calls.append((list(paths), k, final))
        ids = sorted(map(str, paths), key=lambda p: int(Path(p).stem))[:k]
        return Verdict(ids, dict.fromkeys(ids, "best available"))


@pytest.mark.parametrize("n,chunk,k", [(60, 24, 5), (43, 6, 8), (7, 1, 3), (1, 24, 5)])
def test_strongest_chunk_can_supply_all_final_keepers(tmp_path, n, chunk, k):
    photos = photos_in(tmp_path, n)
    args = parse_args([str(tmp_path), "--judge", "--judge-coverage", "full", "--judge-chunk", str(chunk),
                       "--max-per-group", str(k), "--extra-per", "0"])
    oracle = Oracle()
    select_photos(photos, args, oracle)
    assert [p.path.name for p in photos if p.selected] == [p.path.name for p in photos[:k]]
    assert all(p.shortlisted for p in photos)
    assert all(len(paths) <= max(chunk, 2 * k) for paths, _, _ in oracle.calls)


def test_optional_preselection_keeps_low_scoring_scenes_and_expands_if_needed():
    candidates = []
    for i in range(40):
        scene = 0 if i < 33 else i - 32
        candidates.append(ScenedCandidate(str(i), 40 - i, "animal", np.eye(8)[scene],
                                          dino=np.eye(8)[scene], scene=scene))
    for cap in (4, 10):
        selected = set(diverse_preselection(candidates, cap))
        assert {c.scene for c in candidates if c.id in selected} == set(range(8))
        assert len(selected) == max(cap, 8)


def test_distinct_focus_is_never_an_exact_twin_and_subject_focus_beats_background():
    y, x = np.mgrid[:512, :512]
    background = Image.fromarray((80 + ((x // 4 + y // 4) % 2) * 100).astype("uint8")).convert("RGB")
    subject = Image.fromarray((((x[:128, :128] // 12 + y[:128, :128] // 12) % 2)
                               * 180 + 30).astype("uint8")).convert("RGB")
    sharp = background.copy()
    sharp.paste(subject, (192, 192))
    blurred = background.copy()
    blurred.paste(subject.filter(ImageFilter.GaussianBlur(5)), (192, 192))
    assert sharpness(to_gray(sharp)) == sharpness(to_gray(blurred))
    assert float(signature(sharp) @ signature(blurred)) > 0.97
    box = (192 / 512, 192 / 512, 320 / 512, 320 / 512)
    assert subject_focus(sharp, box) > 5 * subject_focus(blurred, box)
    cands = [ScenedCandidate(str(i), i, "bird", np.array([1., 0.]), sig=signature(im), scene=0)
             for i, im in enumerate((sharp, blurred))]
    chunks, twins = chunk_for_tournament(cands)
    assert len(chunks[0]) == 2 and twins == {}


def test_scoring_prefers_subject_focus_and_is_independent_of_other_days_and_labels(tmp_path):
    photos = photos_in(tmp_path, 2)
    for p in photos:
        p.aesthetic = 5
    photos[0].subject_sharpness = 800
    photos[1].subject_sharpness = 20
    args = parse_args([str(tmp_path)])
    score_and_reject(photos, args)
    before = [p.score for p in photos]
    assert before[0] > before[1]
    extras = [Photo(Path(f"extra{i}.jpg"), f"extra{i}", time=datetime(2026, 9, 2),
                    label="bird", aesthetic=9, sharpness=5000) for i in range(20)]
    extras.append(Photo(Path("lion.jpg"), "lion", time=photos[0].time, label="lion", aesthetic=1, sharpness=2))
    score_and_reject(photos + extras, args)
    assert [p.score for p in photos] == before
    assert args.min_score == float("-inf")


def test_judge_sees_difficult_light_and_low_texture_but_respects_skip_labels(tmp_path):
    photos = photos_in(tmp_path, 3)
    photos[0].sharpness = 1
    photos[1].clip_low = 0.8
    photos[2].label = "document or screenshot"
    args = parse_args([str(tmp_path), "--judge"])
    score_and_reject(photos, args)
    select_photos(photos, args, Oracle())
    assert photos[0].shortlisted and photos[1].shortlisted
    assert photos[2].reject == "label" and not photos[2].shortlisted


def test_cross_label_duplicates_are_compared_but_distinct_moments_can_survive(tmp_path):
    photos = photos_in(tmp_path, 2)
    photos[1].label = "eagle"
    photos[1].dino = photos[0].dino
    photos[0].sig = photos[1].sig = np.array([1., 0.])
    args = parse_args([str(tmp_path), "--judge"])

    class Compare(Oracle):
        def judge(self, paths, label, day, k, boxes, final=False):
            return super().judge(paths, label, day, 1 if final else k, boxes, final)

    judge = Compare()
    select_photos(photos, args, judge)
    assert sum(p.selected for p in photos) == 1
    assert any(final and len(paths) == 2 for paths, _, final in judge.calls)
    # A visual judge can distinguish moments that the embedding cannot.
    select_photos(photos, args, Oracle())
    assert all(p.selected for p in photos)


def test_exact_copies_across_labels_only_survive_once(tmp_path):
    photos = photos_in(tmp_path, 2)
    photos[1].label = "eagle"
    photos[0].digest = photos[1].digest = "identical-source"
    args = parse_args([str(tmp_path), "--judge"])
    select_photos(photos, args, Oracle())
    assert sum(p.selected for p in photos) == 1
    assert "identical copy" in photos[1].judge_reason


def test_empty_final_verdict_is_not_replaced_by_score_picks(tmp_path):
    photos = photos_in(tmp_path, 10)
    args = parse_args([str(tmp_path), "--judge", "--judge-chunk", "5", "--max-per-group", "2"])

    class RejectFinal(Oracle):
        def judge(self, paths, label, day, k, boxes, final=False):
            return Verdict([], {}) if final else super().judge(paths, label, day, k, boxes, final)

    select_photos(photos, args, RejectFinal())
    assert not any(p.selected for p in photos)


def test_all_tournament_stages_retry_and_report_fallback(tmp_path, monkeypatch):
    photos = photos_in(tmp_path, 6)
    args = parse_args([str(tmp_path), "--judge", "--judge-chunk", "3", "--max-per-group", "2"])
    monkeypatch.setattr("time.sleep", lambda _: None)

    class FailFinal(Oracle):
        def judge(self, paths, label, day, k, boxes, final=False):
            return Verdict([], {}, error="network") if final else super().judge(paths, label, day, k, boxes, final)

    select_photos(photos, args, FailFinal())
    assert [p.path.name for p in photos if p.selected] == ["000.jpg", "001.jpg"]
    assert all("by score" in p.judge_reason for p in photos if p.selected)


def test_explicit_zero_share_keeps_a_cheaper_budget_based_shortlist(tmp_path, monkeypatch):
    # A user can still choose the smaller shortlist explicitly for a large shoot.
    monkeypatch.setattr(Path, "exists", lambda _: True)
    photos = [Photo(tmp_path / f"{i:04}.jpg", str(i), label="bird", scene=i // 200, score=2000 - i,
                    emb=np.eye(10)[i // 200], dino=np.eye(10)[i // 200]) for i in range(2000)]
    args = parse_args([str(tmp_path), "--judge", "--preselect-share", "0"])
    assert args.judge_coverage == "preselect" and args.preselect_share == 0
    assert args.judge_crops == "single"
    oracle = Oracle()
    select_photos(photos, args, oracle)
    shortlisted = [p for p in photos if p.shortlisted]
    assert len(shortlisted) == 45  # 3 x the capped keeper budget of 15
    assert {p.scene for p in shortlisted} == set(range(10))
    assert sum(len(paths) for paths, _, _ in oracle.calls) == 75  # 45 initial + 30 finalists
    assert sum(p.selected for p in photos) == 15


def test_winner_pools_share_one_final_request_when_they_fit(tmp_path):
    photos = photos_in(tmp_path, 60)
    args = parse_args([str(tmp_path), "--judge", "--judge-coverage", "full", "--extra-per", "0"])
    oracle = Oracle()
    select_photos(photos, args, oracle)
    assert [(len(paths), final) for paths, _, final in oracle.calls] == [
        (20, False), (20, False), (20, False), (15, True)]
    assert [p.path.name for p in photos if p.selected] == [f"{i:03}.jpg" for i in range(5)]


def test_default_percentage_floor_is_half_the_group(tmp_path):
    photos = photos_in(tmp_path, 60)
    for p in photos:
        p.scene = 0
    args = parse_args([str(tmp_path), "--judge", "--extra-per", "0"])
    assert args.preselect_share == 0.5
    select_photos(photos, args, Oracle())
    assert sum(p.shortlisted for p in photos) == 30


@pytest.mark.parametrize("n,share,expected", [(97, 0.5, 49), (197, 0.5, 99), (61, 0.25, 16)])
def test_percentage_floor_rounds_up_to_cover_the_requested_share(tmp_path, n, share, expected):
    photos = photos_in(tmp_path, n)
    for photo in photos:
        photo.scene = 0
    args = parse_args([str(tmp_path), "--judge", "--preselect-share", str(share), "--extra-per", "0"])
    select_photos(photos, args, Oracle())
    assert sum(photo.shortlisted for photo in photos) == expected


def test_broader_default_lets_judge_recover_a_locally_underrated_burst_frame(tmp_path):
    photos = photos_in(tmp_path, 101)
    for photo in photos:
        photo.scene = 0
    favourite = photos[40]

    class MomentJudge(Oracle):
        def judge(self, paths, label, day, k, boxes, final=False):
            self.calls.append((list(paths), k, final))
            ids = sorted(map(str, paths), key=lambda p: (p != str(favourite.path), p))[:k]
            return Verdict(ids, dict.fromkeys(ids, "strong moment"))

    cheap = parse_args([str(tmp_path), "--judge", "--preselect-share", "0"])
    select_photos(photos, cheap, MomentJudge())
    assert not favourite.shortlisted and not favourite.selected
    select_photos(photos, parse_args([str(tmp_path), "--judge"]), MomentJudge())
    assert favourite.shortlisted and favourite.selected

"""Moment recall and local processing budgets, without real models or API calls."""
from datetime import datetime, timedelta

import numpy as np
import pytest
from PIL import Image, ImageFilter

from fotosort import cache
from fotosort.cli import Photo, assign_moments, parse_args, refine_close_focus, score_and_reject, write_report
from fotosort.group import cluster_moments
from fotosort.quality import detail_focus, subject_embedding_crop
from fotosort.selection import Candidate, ScenedCandidate, diverse_preselection, is_duplicate


def unit(values):
    values = np.array(values, dtype=np.float32)
    return values / np.linalg.norm(values)


def test_moments_capture_subject_changes_hidden_by_background():
    whole = np.tile([1., 0.], (5, 1))
    crops = [unit([1, 0]), unit([1, 0]), unit([0, 1]), unit([0, 1]), unit([1, 0])]
    ids, changes = cluster_moments([0, 1, 2, 3, 4], [0] * 5, whole, crops)
    assert ids == [0, 0, 1, 1, 2]
    assert changes[2] == changes[3] and changes[2] > 0
    assert cluster_moments([0, 1, 2, 3, 4], [0] * 5, whole, [None] * 5)[0] == [0] * 5


def test_moments_respect_time_scene_boundaries_and_missing_detections():
    whole = np.tile([1., 0.], (6, 1))
    crops = [unit([1, 0]), None, unit([1, 0]), None, None, None]
    ids, _ = cluster_moments([0, 1, 20, 21, None, None], [0, 0, 0, 1, 1, 1], whole, crops)
    assert ids == [0, 0, 1, 2, 2, 2]
    assert cluster_moments([], [], [], []) == ([], [])


def test_moment_centroid_detects_gradual_drift():
    whole = [unit([np.cos(a), np.sin(a)]) for a in np.linspace(0, 1, 15)]
    ids, _ = cluster_moments(list(range(15)), [0] * 15, whole, [None] * 15)
    assert len(set(ids)) > 1  # adjacent frames alone are all above the threshold


def test_assign_moments_does_not_change_scene_ids_or_cross_days(tmp_path):
    photos = [Photo(tmp_path / f'{i}.jpg', str(i), time=datetime(2026, 9, 1) + timedelta(days=i // 2),
                    label='bird', scene=i // 2, emb=unit([1, 0]), dino=unit([1, 0])) for i in range(4)]
    assign_moments(photos, parse_args([str(tmp_path)]))
    assert photos[0].moment == photos[1].moment != photos[2].moment == photos[3].moment
    assert [p.scene for p in photos] == [0, 0, 1, 1]


def test_quality_aware_coverage_finds_action_without_filling_with_bad_pans():
    # Even time-splitting one repeated pose into many moments must not hide the
    # brief action frame behind eight low-quality, maximally different outliers.
    frames = [ScenedCandidate(f'rest{i}', 3 - i * .01, 'bird', np.eye(10)[0],
                               dino=np.eye(10)[0], scene=0, moment=i) for i in range(20)]
    action = unit([.95, .31225] + [0.] * 8)
    frames.append(ScenedCandidate('action', 2, 'bird', action, dino=action, scene=0, moment=20))
    frames += [ScenedCandidate(f'pan{i}', -8, 'bird', np.eye(10)[i + 2], dino=np.eye(10)[i + 2],
                                scene=0, moment=i + 21) for i in range(8)]
    reasons = {}
    selected = diverse_preselection(frames, 15, reasons=reasons)
    assert len(selected) == 15 and 'action' in selected
    assert sum(i.startswith('pan') for i in selected) == 2
    assert sum(v.startswith('Exploration') for v in reasons.values()) == 2
    assert len(set(selected)) == len(selected)
    assert diverse_preselection(list(reversed(frames)), 15) == selected
    without_exploration = diverse_preselection(frames, 15, explore_slots=0)
    assert 'action' in without_exploration and not any(i.startswith('pan') for i in without_exploration)


def test_subject_comparison_gets_changed_pose_into_fixed_shortlist():
    frames = [ScenedCandidate(str(i), 1 - i * .001, 'bird', unit([1, 0]), dino=unit([1, 0]),
                               subject_dino=unit([1, 0]), scene=0, moment=0) for i in range(30)]
    frames[-1].subject_dino = unit([0, 1])
    frames[-1].moment = 1
    assert '29' in diverse_preselection(frames, 6, explore_slots=0)
    frames[-1].subject_dino = None
    assert '29' not in diverse_preselection(frames, 6, explore_slots=0)  # unknown is not novelty


def test_missing_dino_uses_available_comparisons_without_mixing_model_dimensions():
    frames = [ScenedCandidate(str(i), 1 - i * .01, 'bird', unit([1, 0, 0]),
                               dino=unit([1, 0]) if i else None, scene=0) for i in range(20)]
    frames[-1].dino = unit([0, 1])
    result = diverse_preselection(frames, 6)
    assert len(result) == len(set(result)) == 6 and '19' in result


def test_independent_focus_and_moment_signals_keep_alternatives():
    frames = [ScenedCandidate(str(i), 2 - i * .01, 'bird', unit([1, 0]), scene=0, moment=0,
                               focus=0, distinctiveness=0) for i in range(20)]
    frames[-1].focus = 5
    frames[-2].distinctiveness = .8
    reasons = {}
    selected = diverse_preselection(frames, 12, reasons=reasons)
    assert {'0', '19', '18'} <= set(selected)
    assert reasons['19'] == 'Independent focus nomination'
    assert reasons['18'] == 'Independent distinctiveness nomination'


def test_signal_disagreement_can_use_a_bounded_exploration_slot():
    # The signal champion already seeds the scene; retain a lower-scoring
    # alternative with strong conflicting evidence over an ordinary runner-up.
    frames = [ScenedCandidate(str(i), 2 - i * .01, 'bird', unit([1, 0]), scene=0,
                               aesthetic=1 if i == 0 else 0) for i in range(20)]
    frames[-1].aesthetic = .99
    reasons = {}
    selected = diverse_preselection(frames, 6, explore_slots=1, reasons=reasons)
    assert '19' in selected and reasons['19'].startswith('Exploration')
    assert sum(r.startswith('Exploration') for r in reasons.values()) == 1


@pytest.mark.parametrize('cap,scenes', [(1, 1), (1, 8), (10, 4), (18, 10)])
def test_moment_count_and_exploration_never_expand_scene_adjusted_cap(cap, scenes):
    frames = [ScenedCandidate(str(i), 50 - i, 'bird', unit([1, 0]), scene=i % scenes,
                               moment=i) for i in range(50)]
    result = diverse_preselection(frames, cap, explore_slots=100)
    assert len(result) == len(set(result)) == max(cap, scenes)
    assert {p.scene for p in frames if p.id in result} == set(range(scenes))


def test_subject_difference_prevents_local_duplicate_veto_but_not_exact_copy():
    a = Candidate('a', 1, 'bird', unit([1, 0]), dino=unit([1, 0]), subject_dino=unit([1, 0]))
    b = Candidate('b', 1, 'bird', unit([1, 0]), dino=unit([1, 0]), subject_dino=unit([0, 1]))
    assert not is_duplicate(a, [b], .89, .9)
    a.digest = b.digest = 'same-bytes'
    assert is_duplicate(a, [b], .89, .9)


def focus_photos(tmp_path, moments=4):
    return [Photo(tmp_path / f'{i}.jpg', str(i), label='bird', scene=0, moment=i // 2,
                  box=(.25, .25, .75, .75), score=2 - i // 2 * .1, aesthetic=5, sharpness=100,
                  emb=unit([1, 0]), dino=unit([1, 0]), subject_sharpness=100) for i in range(moments * 2)]


def test_adaptive_focus_budget_source_mapping_and_cached_reuse(tmp_path, monkeypatch):
    photos = focus_photos(tmp_path)
    photos[1].source = tmp_path / 'developed.dng'
    calls = []

    def measure(path, box):
        calls.append(path)
        return 1000 if path.endswith('developed.dng') else 20

    monkeypatch.setattr('fotosort.cli.detail_focus', measure)
    args = parse_args([str(tmp_path), '--judge', '--no-cache', '--focus-refine-per-group', '4'])
    refine_close_focus(photos, tmp_path, args)
    assert len(calls) == 4 and str(photos[1].source) in calls
    assert photos[1].score > photos[0].score
    assert sum(bool(p.focus_refinement) for p in photos) == 4
    assert all(abs(p.refined_focus_adjustment) <= 2 * (1 - args.aesthetic_weight) for p in photos)
    score_and_reject(photos, args)
    refine_close_focus(photos, tmp_path, args)
    assert len(calls) == 4  # cached on the Photo; disk cache coverage lives in pipeline tests


def test_straightforward_or_disabled_focus_choices_do_not_decode(tmp_path, monkeypatch):
    photos = focus_photos(tmp_path, 1)
    photos[1].score = -5
    monkeypatch.setattr('fotosort.cli.detail_focus', lambda *_: pytest.fail('unexpected full decode'))
    args = parse_args([str(tmp_path), '--no-cache'])
    refine_close_focus(photos, tmp_path, args)
    photos[1].score = photos[0].score
    args.focus_refine_per_group = 0
    refine_close_focus(photos, tmp_path, args)
    args.focus_refine_per_group = 1
    refine_close_focus(photos, tmp_path, args)


def test_refined_focus_cache_is_reused_on_a_fresh_run(tmp_path, monkeypatch):
    photos = focus_photos(tmp_path, 1)
    for ph in photos:
        ph.sig = unit([1, 0])
    calls = []

    def measure(path, box):
        calls.append(path)
        return 100 if path.endswith('0.jpg') else 1000

    monkeypatch.setattr('fotosort.cli.detail_focus', measure)
    args = parse_args([str(tmp_path), '--judge'])
    refine_close_focus(photos, tmp_path, args)
    assert len(calls) == 2
    loaded = cache.load(tmp_path, args.detector)
    again = focus_photos(tmp_path, 1)
    for ph in again:
        for name, value in loaded[ph.key].items():
            setattr(ph, name, value)
    refine_close_focus(again, tmp_path, args)
    assert len(calls) == 2
    assert [p.score for p in again] == [p.score for p in photos]


@pytest.mark.parametrize('failure', [None, OSError('bad source')])
def test_incomplete_focus_comparison_never_mixes_preview_and_detail(tmp_path, monkeypatch, failure):
    photos = focus_photos(tmp_path, 1)
    before = [p.score for p in photos]

    def measure(path, box):
        if path.endswith('0.jpg'):
            return 100
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr('fotosort.cli.detail_focus', measure)
    refine_close_focus(photos, tmp_path, parse_args([str(tmp_path), '--no-cache']))
    assert [p.score for p in photos] == before
    assert photos[1].focus_refinement


def test_full_detail_focus_sees_sharp_subject_and_skips_tiny_regions(tmp_path):
    y, x = np.mgrid[:1200, :1200]
    img = Image.fromarray((((x // 6 + y // 6) % 2) * 160 + 40).astype('uint8')).convert('RGB')
    good, bad = tmp_path / 'good.jpg', tmp_path / 'bad.jpg'
    img.save(good)
    img.filter(ImageFilter.GaussianBlur(5)).save(bad)
    box = (.25, .25, .75, .75)
    assert detail_focus(str(good), box) > 5 * detail_focus(str(bad), box)
    assert detail_focus(str(good), (0, 0, .01, .01)) is None
    assert subject_embedding_crop(img, None) is None
    assert subject_embedding_crop(img, (0, 0, .001, .001)) is None
    assert subject_embedding_crop(img, box).size == (600, 600)


def test_report_explains_moments_and_focus_refinement(tmp_path):
    import csv

    photos = focus_photos(tmp_path, 1)
    photos[0].subject_dino = unit([1, 0])
    photos[0].preselection_reason = 'Independent focus nomination'
    photos[0].detail_focus = 123
    photos[0].refined_focus_adjustment = .1
    write_report(photos, tmp_path / 'report.csv', tmp_path)
    with (tmp_path / 'report.csv').open() as stream:
        row = next(csv.DictReader(stream))
    assert row['moment'] == '0' and row['subject_embedding'] == '1'
    assert row['detail_focus'] == '123.0000' and row['refined_focus_adjustment'] == '0.1000'
    assert row['preselection_reason'] == 'Independent focus nomination'


@pytest.mark.parametrize('flag,value', [('--moment-sim', 'nan'), ('--moment-sim', '1.1'),
                                       ('--moment-gap-seconds', '0'), ('--moment-gap-seconds', 'inf'),
                                       ('--preselect-explore', '-1'), ('--focus-refine-per-group', '-1')])
def test_invalid_local_selection_settings(flag, value):
    with pytest.raises(SystemExit):
        parse_args(['/tmp', flag, value])

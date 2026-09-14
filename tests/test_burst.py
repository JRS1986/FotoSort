"""General burst behaviour, including non-wildlife cases; no model/API calls."""
import math
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from fotosort import cache
from fotosort.burst import BURST_PROMPTS, burst_scores, interaction_box, local_burst_views, valid_boxes
from fotosort.cli import Photo, _candidate, assign_moments, parse_args
from fotosort.group import cluster_bursts
from fotosort.selection import ScenedCandidate, chunk_for_tournament, diverse_preselection


def unit(*values):
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def frames(n=30, bucket='portrait'):
    return [ScenedCandidate(str(i), 3 - i * .01, bucket, unit(1, 0), dino=unit(1, 0),
                             scene=0, burst=i // 6, capture_time=i, moment=i // 6) for i in range(n)]


def test_bursts_keep_expression_changes_but_split_capture_gaps_and_views():
    vectors = [unit(1, 0), unit(.9, .3), unit(1, 0), unit(0, 1), unit(0, 1), unit(0, 1)]
    assert cluster_bursts([0, 1, 5, 6, 7, 8], [0, 0, 0, 0, 1, 1], vectors) == [0, 0, 1, 2, 3, 3]
    assert cluster_bursts([0, 3, 6, 9, 12], [0] * 5, [unit(1, 0)] * 5) == [0, 0, 0, 0, 1]


def test_missing_or_reversed_times_do_not_invent_a_burst():
    assert cluster_bursts([None, 10, float('nan'), 11, 9], [0] * 5, [unit(1, 0)] * 5) == [-1, 0, -1, 1, 2]
    assert cluster_bursts([], [], []) == []


def test_interaction_view_includes_small_secondary_subject_and_is_bounded():
    adult, small = (.3, .3, .5, .8), (.6, .6, .7, .75)
    region = interaction_box(adult, [adult, small])
    assert region[0] < adult[0] and region[2] > small[2]
    assert region[1] < adult[1] and region[3] > adult[3]
    views = local_burst_views(Image.new('RGB', (1024, 768)), adult, [adult, small])
    assert set(views) == {'subject', 'interaction'}
    assert views['interaction'].width > views['subject'].width
    assert not local_burst_views(Image.new('RGB', (1024, 768)), None, [])
    assert not local_burst_views(Image.new('RGB', (1024, 768)), (0, 0, 1, 1), [])
    assert len(valid_boxes([adult] * 10 + [(-1, 0, 1, 1), (0, 0, float('nan'), 1)])) == 1


def test_detector_retains_secondary_boxes_without_changing_primary_metrics():
    pytest.importorskip("torch")  # CI installs only the light test dependencies
    from fotosort.embed import SubjectDetector

    detections = [SimpleNamespace(cls=cls, conf=.9, xyxy=np.array([box])) for cls, box in [
        (0, [100, 100, 500, 600]), (1, [520, 400, 600, 500]),
        (2, [100, 700, 200, 900]), (3, [0, 0, 1000, 1000])]]
    result = SimpleNamespace(orig_shape=(1000, 1000), boxes=detections,
                             names={0: 'dog', 1: 'cat', 2: 'person', 3: 'car'})
    detector = object.__new__(SubjectDetector)
    detector.device, detector.conf = 'cpu', .3
    detector.model = SimpleNamespace(predict=lambda *a, **kw: [result])
    measured = detector.detect([Image.new('RGB', (1000, 1000))])[0]
    assert measured['subject'] == pytest.approx(.2) and measured['person'] == pytest.approx(.02)
    assert measured['box'] == (.1, .1, .5, .6) and not measured['edge']
    assert (.52, .4, .6, .5) in measured['boxes'] and len(measured['boxes']) == 3


def test_story_scores_use_secondary_view_without_a_species_or_motion_rule():
    texts = np.array([[1., 0], [0, 0], [0, 1], [0, 0], [.5, .5], [0, 0]])
    whole, interaction = unit(1, 0), unit(0, 1)
    scores = burst_scores(whole, None, interaction, texts)
    assert scores['expression'] == scores['interaction'] == 1
    assert burst_scores(whole, None, None, texts)['interaction'] == 0
    assert set(scores) == set(BURST_PROMPTS)


@pytest.mark.parametrize('bucket', ['portrait', 'bird', 'landscape', 'architecture'])
def test_burst_capacity_preserves_poses_without_semantic_predictions(bucket):
    candidates = frames(bucket=bucket)
    # A changed pose in a continuous sequence has only a modest overall score.
    candidates[16].interaction_dino = unit(0, 1)
    for i in [12, 13, 14, 15, 17]:
        candidates[i].interaction_dino = unit(1, 0)
    reasons = {}
    chosen = diverse_preselection(candidates, 15, reasons=reasons)
    assert len(chosen) == 15 and len(set(chosen)) == 15
    assert any(reason.startswith('Burst') for reason in reasons.values())
    assert '16' in chosen
    assert sum(reason in ['Protected quality allocation', 'Scene representative'] for reason in reasons.values()) >= 6
    assert diverse_preselection(list(reversed(candidates)), 15) == chosen


def test_burst_semantics_are_local_and_do_not_require_a_high_overall_score():
    candidates = frames()
    # Expression is informative within this burst, even for its weakest frame.
    for i in range(12, 18):
        candidates[i].burst_signals = {'expression': .01}
    candidates[17].burst_signals = {'expression': .05}
    candidates[17].score = -2
    reasons = {}
    chosen = diverse_preselection(candidates, 15, reasons=reasons)
    assert '17' in chosen and reasons['17'] == 'Burst alternative: expression'
    assert candidates[17].score == -2  # nomination did not train or rescore anything


def test_flat_or_missing_semantic_signals_do_not_invent_expression_evidence():
    candidates = frames()
    for candidate in candidates:
        candidate.burst_signals = {'expression': .01 + float(candidate.id) * .00001,
                                   'interaction': float('nan')}
    reasons = {}
    diverse_preselection(candidates, 15, reasons=reasons)
    assert not any('alternative: expression' in r or 'alternative: interaction' in r for r in reasons.values())


def test_missing_visual_features_are_not_a_novel_pose():
    candidates = frames()
    candidates[19].dino = None
    candidates[19].score = -5
    reasons = {}
    diverse_preselection(candidates, 20, reasons=reasons)
    assert reasons.get('19') != 'Burst pose alternative'


def test_default_does_not_use_cached_experimental_signals(tmp_path):
    photo = Photo(tmp_path/'portrait.jpg', 'portrait', time=datetime(2026, 9, 1),
                  emb=unit(1, 0), dino=unit(1, 0), label='people', scene=0,
                  interaction_dino=unit(0, 1), burst_signals={'expression': .5}, burst_features_checked=True)
    args = parse_args([str(tmp_path), '--judge'])
    assert args.preselect_share == .5 and not args.burst_alternatives
    assign_moments([photo], args)
    candidate = _candidate(photo)
    assert candidate.burst == -1 and candidate.interaction_dino is None and not candidate.burst_signals
    args.burst_alternatives = True
    assign_moments([photo], args)
    assert _candidate(photo).burst >= 0 and _candidate(photo).burst_signals['expression'] == .5


@pytest.mark.parametrize('n,chunk', [(60, 24), (39, 20), (52, 24), (101, 16)])
def test_burst_packing_never_increases_requests_or_repeats_frames(n, chunk):
    candidates = frames(n)
    for i, candidate in enumerate(candidates):
        candidate.burst = i // 13
    batches, twins = chunk_for_tournament(candidates, chunk)
    assert not twins
    assert len(batches) == math.ceil(n / chunk)
    assert all(0 < len(batch) <= chunk for batch in batches)
    assert sorted(i for batch in batches for i in batch) == sorted(c.id for c in candidates)
    assert chunk_for_tournament(list(reversed(candidates)), chunk)[0] == batches


def test_small_burst_alternatives_stay_in_the_same_request_when_they_fit():
    candidates = frames(24)
    batches, _ = chunk_for_tournament(candidates, 12)
    for start in range(0, 24, 6):
        assert any({str(i) for i in range(start, start + 6)} <= set(batch) for batch in batches)


def test_burst_features_cache_round_trip(tmp_path):
    entry = dict(emb=unit(1, 0), dino=unit(1, 0), sig=unit(1, 0), person=0, subject=.2,
                 edge=False, sharpness=100, clip_low=0, clip_high=0, mean=100,
                 subject_clip=unit(0, 1), interaction_clip=unit(1, 1), interaction_dino=unit(0, 1),
                 subject_boxes=[(.1, .2, .3, .4), (.4, .5, .6, .7)],
                 subject_boxes_checked=True, burst_features_checked=True)
    cache.save(tmp_path, {'image': entry})
    loaded = cache.load(tmp_path)['image']
    for field in ['subject_clip', 'interaction_clip', 'interaction_dino']:
        assert np.allclose(loaded[field], entry[field])
    assert np.allclose(loaded['subject_boxes'], entry['subject_boxes'])
    assert loaded['subject_boxes_checked'] and loaded['burst_features_checked']

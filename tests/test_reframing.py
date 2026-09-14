"""Reframing must not defeat strongly supported local duplicate matches."""
import numpy as np
import pytest

from fotosort.cli import parse_args
from fotosort.selection import ScenedCandidate, chunk_for_tournament, is_duplicate, select_day


def matching_pair(full, crop, pixel, same_scene=True):
    def vector(sim):
        return np.array([sim, np.sqrt(1 - sim ** 2)])
    a = ScenedCandidate('a', 2, 'giraffe', np.array([1., 0.]), sig=np.array([1., 0.]),
                         dino=np.array([1., 0.]), subject_dino=np.array([1., 0.]), scene=1)
    b = ScenedCandidate('b', 1, 'giraffe', np.array([1., 0.]), sig=vector(pixel),
                         dino=vector(full), subject_dino=vector(crop), scene=1 if same_scene else 2)
    return a, b


@pytest.mark.parametrize('full,crop,pixel', [(.9456, .9162, .6941), (.9649, .9639, .1880),
                                            (.9910, .9834, .8457)])
def test_reviewed_similarity_measurements_allow_reframing(full, crop, pixel):
    a, b = matching_pair(full, crop, pixel)
    assert is_duplicate(a, [b], .89, .9, reframe_sim=.94)
    assert not is_duplicate(a, [b], .89, .9, reframe_sim=0)
    assert select_day([a, b], {'giraffe': 2}, .89, .9, reframe_sim=.94) == ['a']
    # The visual judge still receives both non-identical frames.
    assert chunk_for_tournament([a, b])[0] == [['a', 'b']]


def test_default_preserves_moderate_reframing_alternatives_but_merges_almost_identical_pair():
    # A 0.94 trial merged these poses, then the heuristic chose the user's
    # less-preferred frame. Default merging must preserve these alternatives.
    a, b = matching_pair(.9649, .9639, .188)
    assert not is_duplicate(a, [b], .89, .9)
    a, b = matching_pair(.991, .9834, .8457)
    assert is_duplicate(a, [b], .89, .9)
    assert parse_args(['/tmp']).dup_reframe_sim == .98


def test_background_alone_missing_detections_or_different_context_cannot_bypass_pixels():
    a, b = matching_pair(.99, .7, .2)
    assert not is_duplicate(a, [b], .89, .9)
    a, b = matching_pair(.99, .99, .2, same_scene=False)
    assert not is_duplicate(a, [b], .89, .9)
    b.scene = a.scene
    b.bucket = 'different subject'
    assert not is_duplicate(a, [b], .89, .9)
    b.bucket = a.bucket
    b.subject_dino = None
    assert not is_duplicate(a, [b], .89, .9)
    a.scene = b.scene = -1
    assert not is_duplicate(a, [b], .89, .9)


def test_user_duplicate_threshold_still_controls_reframed_matches():
    a, b = matching_pair(.95, .93, .2)
    assert not is_duplicate(a, [b], .97, .9)
    assert not is_duplicate(a, [b], .89, .9, reframe_sim=.99)


@pytest.mark.parametrize('value', ['-0.1', '1.1', 'nan'])
def test_reframing_threshold_validation(value):
    with pytest.raises(SystemExit):
        parse_args(['/tmp', '--dup-reframe-sim', value])

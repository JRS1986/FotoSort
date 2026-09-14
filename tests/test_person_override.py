from pathlib import Path

from fotosort.cli import Photo, apply_person_override


def _p(label, person):
    return Photo(path=Path("x.jpg"), key="k", label=label, label_prob=0.5, person=person)


def test_visible_person_overrides_animal_label():
    photos = [_p("kudu antelope", 0.20), _p("kudu antelope", 0.0), _p("landscape", 0.005)]
    apply_person_override(photos, 0.02)
    assert [p.label for p in photos] == ["people", "kudu antelope", "landscape"]


def test_override_disabled():
    photos = [_p("kudu antelope", 0.5)]
    apply_person_override(photos, 0)
    assert photos[0].label == "kudu antelope"


def test_detector_false_positive_needs_clip_agreement():
    seal = _p("seal swimming in the water", 0.17)
    seal.people_prob = 0.006
    person = _p("seal swimming in the water", 0.12)
    person.people_prob = 0.36
    big = _p("seal swimming in the water", 0.68)
    big.people_prob = 0.06
    apply_person_override([seal, person, big], 0.02, 0.2)
    assert seal.label == "seal swimming in the water"
    assert person.label == "people"
    assert big.label == "people"  # a person filling most of the frame is trusted

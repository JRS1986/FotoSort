from fotosort.editing import PRESETS, STYLE_TO_PRESET, benefit_band, edit_benefit
from fotosort.judge import parse_verdict


def test_edit_benefit_rewards_recoverable_problems():
    assert edit_benefit(0.0, 0.0, 120, 55, 200) < 10                    # clean daylight JPEG
    assert edit_benefit(0.06, 0.0, 130, 55, 200) >= 45                   # blown sky
    assert edit_benefit(0.0, 0.2, 60, 40, 6400) >= 50                    # dark, crushed, noisy -> high
    assert benefit_band(10) == "low" and benefit_band(30) == "medium" and benefit_band(70) == "high"
    assert set(STYLE_TO_PRESET.values()) <= set(PRESETS)


def test_verdict_carries_edit_benefit_and_preset():
    v = parse_verdict('{"picks": [{"photo": 1, "reason": "r", "edit_benefit": "High", "edit_why": "blown sky", '
                      '"preset": "color pop"}, {"photo": 2, "reason": "r2"}]}', ["a", "b"])
    assert v.picks == ["a", "b"]
    assert v.edits["a"] == {"benefit": "high", "why": "blown sky", "preset": "Color Pop"}
    assert "b" not in v.edits

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def rightwayup_stub(monkeypatch):
    """A deterministic model stand-in; tests never download weights."""
    class Orienter:
        instances = []
        angle = 90.0
        abstain = False
        precision = "int8"
        device = "CPUExecutionProvider"

        def __init__(self, tier, abstain):
            assert abstain == "strict"
            self.tier = tier
            self.seen = []
            self.results = []
            self.corrections = []
            self.instances.append(self)

        def predict(self, image):
            self.seen.append(np.array(image))
            result = SimpleNamespace(angle_cw=self.angle, confidence=0.9, abstain=self.abstain,
                                     tier=self.tier, routed=False)
            self.results.append(result)
            return result

        def correct(self, image, *, snap, result):
            assert result is self.results[-1]  # integration must reuse the prediction
            self.corrections.append((snap, result))
            if result.abstain:
                return image.copy()
            angle = round(result.angle_cw / snap) * snap if snap else result.angle_cw
            return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

    monkeypatch.setitem(sys.modules, "rightwayup", SimpleNamespace(Orienter=Orienter))
    return Orienter

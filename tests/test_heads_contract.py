"""Regression tests for the head-loading contract.

Every consumer has to build its text input from the checkpoint's `text_fields`
list instead of assuming a particular encoding. A hardcoded `fx.text_ctx` in the
counterfactual evaluation broke silently the moment the fused head started
consuming two text encodings, and it only showed up as a LayerNorm shape error
buried in a 40-line traceback. These tests make that kind of bug loud and
immediate.

Skipped cleanly if the heads haven't been trained yet.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from lelamp.config import HEADS, N_EMO
from lelamp.train import CONFIGS

pytestmark = pytest.mark.skipif(
    not (HEADS / "fused.pt").exists(),
    reason="heads not trained yet; run `make train`")


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_every_head_loads_and_scores_its_own_inputs(name):
    from lelamp.data.features import load_split
    from lelamp.evaluate.metrics import load_head, predict_probs

    if not (HEADS / f"{name}.pt").exists():
        pytest.skip(f"{name} not trained")
    m, ck = load_head(name)
    fx = load_split("dev")
    p = predict_probs(m, ck, fx)
    assert p.shape == (len(fx), N_EMO)
    assert np.allclose(p.sum(1), 1.0, atol=1e-4), "probabilities must be normalised"
    assert (p >= 0).all()


def test_text_input_width_matches_checkpoint_field_list():
    """The exact invariant the counterfactual bug violated."""
    from lelamp.data.features import load_split
    from lelamp.evaluate.metrics import load_head
    from lelamp.models.text import TEXT_DIM

    m, ck = load_head("fused")
    fx = load_split("dev")
    text = np.concatenate([fx.__dict__[f] for f in ck["text_fields"]], 1)
    assert text.shape[1] == TEXT_DIM * ck["n_text_fields"]
    assert text.shape[1] + fx.vision.shape[1] == m.in_dim


def test_gate_recovers_text_only_path_when_driven_to_zero():
    """The whole point of gated fusion: g -> 0 must reproduce the text head."""
    from lelamp.models.heads import GatedFusionHead

    h = GatedFusionHead(n_text_fields=2).eval()
    t = torch.randn(4, 1536)
    v = torch.randn(4, 2048)
    with torch.no_grad():
        torch.nn.init.constant_(h.gate[-1].bias, -50.0)  # force sigmoid -> 0
        torch.nn.init.zeros_(h.gate[-1].weight)
        fused = h(t, v)
        text_only = h.text_head(t)
    assert torch.allclose(fused, text_only, atol=1e-5)


def test_calibration_temperature_is_applied():
    from lelamp.evaluate.metrics import load_head, predict_probs
    from lelamp.data.features import load_split

    m, ck = load_head("fused")
    fx = load_split("dev")
    cal = predict_probs(m, ck, fx, calibrated=True)
    raw = predict_probs(m, ck, fx, calibrated=False)
    if abs(ck["temperature"] - 1.0) > 1e-3:
        assert not np.allclose(cal, raw), "temperature had no effect"

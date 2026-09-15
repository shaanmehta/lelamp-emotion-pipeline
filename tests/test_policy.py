"""The behaviour mapping is the contract someone writing motion code builds
against, so I assert it instead of eyeballing it. These tests caught two real
bugs while I was building: hue interpolating through green on the way from calm
to angry, and the prior de-bias flattening every arousal region into `idle`.
"""
from __future__ import annotations

import numpy as np

from lelamp.config import EMOTIONS, TAU_ABSTAIN
from lelamp.runtime.policy import decide, prior_offset, project_va

# MELD train class counts -- the zero point for "no information".
TRAIN_COUNTS = np.array([4710, 1743, 683, 1109, 1205, 268, 271], dtype=float)
OFF = prior_offset(TRAIN_COUNTS)

# What a realistically confident head emits: 0.75 on the winner, rest spread.
def peaked(emotion: str, p_max: float = 0.75) -> np.ndarray:
    p = np.full(len(EMOTIONS), (1 - p_max) / (len(EMOTIONS) - 1))
    p[EMOTIONS.index(emotion)] = p_max
    return p


EXPECTED = {
    "neutral":  {"idle", "acknowledge"},
    "joy":      {"celebrate"},
    "sadness":  {"soothe"},
    "anger":    {"recoil"},
    "surprise": {"startle"},
    "fear":     {"startle"},
    "disgust":  {"recoil"},
}


def test_pure_emotions_land_in_intended_regions():
    for emo, ok in EXPECTED.items():
        p = peaked(emo)
        v, a = project_va(p, OFF)
        b = decide(p, float(p.max()), v, a)
        assert b.intent in ok, f"{emo}: v={v:+.2f} a={a:+.2f} -> {b.intent}, want {ok}"


def test_low_confidence_never_commits():
    """A flat posterior must not produce a committed emotional display."""
    p = np.full(7, 1 / 7)
    v, a = project_va(p, OFF)
    b = decide(p, float(p.max()), v, a)
    assert b.intent in {"attend", "confused_tilt"}
    assert b.expressivity < 0.15


def test_disagreement_routes_to_tilt():
    p = peaked("anger", 0.40)
    b = decide(p, 0.40, *project_va(p, OFF), agreement=0.2, tau=TAU_ABSTAIN)
    assert b.intent == "confused_tilt"


def test_hue_takes_the_short_arc_not_through_green():
    """Escalating negative arousal must not render as 'turning green'."""
    hues = []
    for pm in (0.3, 0.5, 0.7, 0.9):
        p = peaked("anger", pm)
        hues.append(decide(p, pm, *project_va(p, OFF)).light.hue_deg)
    # Path must stay out of the green band (70..170 deg) entirely.
    assert all(not (70 < h < 170) for h in hues), hues


def test_expressivity_is_monotone_in_confidence():
    prev = -1.0
    for pm in (0.2, 0.4, 0.6, 0.8, 0.95):
        p = peaked("joy", pm)
        e = decide(p, pm, *project_va(p, OFF)).expressivity
        assert e >= prev, (pm, e, prev)
        prev = e

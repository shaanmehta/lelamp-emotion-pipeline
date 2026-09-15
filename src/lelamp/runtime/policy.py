"""Belief -> behaviour. A pure function, zero learned parameters.

Two properties this is designed to have:

1. It consumes CONFIDENCE, not argmax. Low confidence is routed to `attend` /
   `confused_tilt` -- "I'm listening, I don't know yet" is a legitimate and
   often correct thing for a lamp to express. Expressivity is a continuous gain
   on confidence, so the lamp's commitment is visibly proportional to the
   model's.

2. It is driven by the continuous (valence, arousal) projection rather than the
   emotion label, so a posterior split 0.45 anger / 0.40 disgust produces a
   coherent in-between behaviour instead of snapping to whichever won by 0.05.

Valence/arousal are a DETERMINISTIC PROJECTION of the posterior onto fixed
circumplex coordinates, de-biased by the training prior so that "posterior ==
class prior" maps to (0, 0), i.e. no information reads as neutral rather than
as mildly-negative. MELD has no VA labels; we do not learn these.
"""
from __future__ import annotations

import numpy as np

from ..config import EMOTIONS, TAU_ABSTAIN, VA_COORDS, VA_SCALE
from ..schema import Behavior, Light, Motion

VA_MATRIX = np.array([VA_COORDS[e] for e in EMOTIONS], dtype=np.float64)  # (7,2)


def project_va(probs: np.ndarray, offset: np.ndarray | None = None) -> tuple[float, float]:
    p = np.asarray(probs, dtype=np.float64)
    va = p @ VA_MATRIX
    if offset is not None:
        va = va - offset
    va = va / np.asarray(VA_SCALE)
    return float(np.clip(va[0], -1, 1)), float(np.clip(va[1], -1, 1))


def prior_offset(label_counts: np.ndarray) -> np.ndarray:
    """VA of the class prior -- the zero point for "no information"."""
    pi = label_counts / label_counts.sum()
    return pi @ VA_MATRIX


def _lerp(a, b, t):
    return a + (b - a) * float(np.clip(t, 0, 1))


def _lerp_hue(a: float, b: float, t: float) -> float:
    """Interpolate around the colour wheel the SHORT way.

    Interpolating hue linearly in degrees is a classic bug: 200 deg (cyan) to
    4 deg (red) passes straight through 100 deg (green), so "getting angrier"
    would render as "turning green". Take the shorter arc instead.
    """
    d = ((b - a + 180.0) % 360.0) - 180.0
    return (a + d * float(np.clip(t, 0, 1))) % 360.0


def decide(
    probs: np.ndarray,
    confidence: float,
    valence: float,
    arousal: float,
    agreement: float = 1.0,
    tau: float = TAU_ABSTAIN,
) -> Behavior:
    """Map belief -> a complete motion + light contract."""
    a_n = (arousal + 1) / 2                      # 0..1 arousal
    mag = float(min(1.0, (valence ** 2 + arousal ** 2) ** 0.5))
    expressivity = float(np.clip((confidence - 0.15) / 0.75, 0.05, 1.0))

    # --- intent selection -------------------------------------------------
    if confidence < tau:
        # Uncertain. If the modalities also disagree, say so with the body:
        # a tilt reads as "I'm not sure", which is more honest than a guess.
        intent = "confused_tilt" if agreement < 0.5 else "attend"
        priority = 1
    elif mag < 0.35:
        intent = "acknowledge" if a_n > 0.45 else "idle"
        priority = 1 if intent == "acknowledge" else 0
    elif valence >= 0.15:
        intent = "celebrate" if arousal > 0.25 else "acknowledge"
        priority = 2 if intent == "celebrate" else 1
    elif valence <= -0.15:
        if arousal > 0.25:
            # High-arousal negative. Surprise-flavoured => startle;
            # anger/disgust-flavoured => recoil.
            surp = float(probs[EMOTIONS.index("surprise")] + probs[EMOTIONS.index("fear")])
            aggr = float(probs[EMOTIONS.index("anger")] + probs[EMOTIONS.index("disgust")])
            intent, priority = ("startle", 3) if surp > aggr else ("recoil", 3)
        else:
            intent, priority = "soothe", 2
    else:
        intent, priority = ("startle", 3) if arousal > 0.55 else ("attend", 1)

    # --- continuous parameters -------------------------------------------
    # Hue runs warm-amber (positive) -> desaturated cool (neutral) -> red
    # (negative & aroused) / deep blue (negative & calm).
    if valence >= 0:
        hue = _lerp_hue(200.0, 48.0, valence / 0.6)    # cool -> warm amber
    elif arousal > 0.15:
        hue = _lerp_hue(200.0, 4.0, -valence / 0.8)    # cool -> blue/violet -> red
    else:
        hue = _lerp_hue(200.0, 232.0, -valence / 0.8)  # cool -> deep blue

    sat = float(np.clip(0.18 + 0.70 * mag * expressivity, 0.0, 1.0))
    value = float(np.clip(0.28 + 0.50 * a_n * (0.4 + 0.6 * expressivity), 0.05, 1.0))
    pulse = 0.0
    if intent in ("celebrate", "startle"):
        pulse = round(float(1.2 + 1.6 * a_n), 2)
    elif intent == "soothe":
        pulse = 0.35

    posture, gaze = {
        "idle":          ("upright", "away"),
        "attend":        ("lean_in", "speaker"),
        "acknowledge":   ("upright", "speaker"),
        "celebrate":     ("lean_in", "speaker"),
        "soothe":        ("droop", "speaker"),
        "recoil":        ("lean_back", "speaker"),
        "startle":       ("recoil_pose", "speaker"),
        "confused_tilt": ("upright", "scan"),
    }[intent]

    amplitude = float(np.clip((0.15 + 0.75 * mag) * expressivity, 0.03, 1.0))
    speed = float(np.clip(0.20 + 0.75 * a_n, 0.05, 1.0))

    # Calm states are held long and released slowly; sharp states are brief.
    hold_ms = int(_lerp(2200, 600, a_n))
    decay_ms = int(_lerp(1600, 450, a_n))
    if intent in ("idle", "attend"):
        hold_ms, decay_ms = 1500, 900

    return Behavior(
        intent=intent,
        priority=priority,
        hold_ms=hold_ms,
        decay_ms=decay_ms,
        expressivity=round(expressivity, 3),
        light=Light(round(hue, 1), round(sat, 3), round(value, 3), pulse),
        motion=Motion(posture, gaze, round(amplitude, 3), round(speed, 3)),
    )

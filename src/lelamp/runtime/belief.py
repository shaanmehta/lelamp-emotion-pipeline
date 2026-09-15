"""Emotional inertia.

Turns a stream of noisy per-utterance posteriors into a belief a physical object
can act on without looking broken. Three mechanisms, all cheap, none of them
learned:

1. EMA on the probability simplex. Averages evidence, not decisions.
2. Margin hysteresis. A challenger has to beat the incumbent by
   HYSTERESIS_MARGIN in the smoothed posterior, not just tie it.
3. Minimum dwell. The incumbent is immune for MIN_DWELL_MS after taking over.

Whether this raises F1 is not the only thing to measure. It does raise F1 on
MELD, partly because MELD emotions are sticky inside a dialogue, and that's a
property of the dataset rather than evidence the lamp looks better. So I also
report switch rate in switches per minute, which is what the behaviour layer
actually pays for. See evaluate/smoothing.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import EMA_ALPHA, EMOTIONS, HYSTERESIS_MARGIN, MIN_DWELL_MS


@dataclass
class BeliefOut:
    probs: np.ndarray       # smoothed posterior
    emotion: str            # committed (post-hysteresis) label
    raw_emotion: str        # argmax of the incoming, unsmoothed posterior
    prev_emotion: str | None
    switched: bool
    dwell_ms: int           # how long the *previous* label had been held
    suppressed: int         # cumulative switches hysteresis has eaten


class BeliefTracker:
    """One tracker per dialogue. Reset between dialogues -- carrying belief
    across a conversation boundary is a bug, not inertia."""

    def __init__(self, alpha: float = EMA_ALPHA, margin: float = HYSTERESIS_MARGIN,
                 min_dwell_ms: int = MIN_DWELL_MS):
        self.alpha, self.margin, self.min_dwell_ms = alpha, margin, min_dwell_ms
        self.reset()

    def reset(self):
        self.ema: np.ndarray | None = None
        self.current: str | None = None
        self.last_switch_ms: float = 0.0
        self.suppressed = 0

    def update(self, probs: np.ndarray, t_ms: float) -> BeliefOut:
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / max(probs.sum(), 1e-9)
        raw = EMOTIONS[int(probs.argmax())]

        self.ema = probs if self.ema is None else (
            self.alpha * probs + (1 - self.alpha) * self.ema
        )
        sm = self.ema / self.ema.sum()

        challenger = EMOTIONS[int(sm.argmax())]
        prev = self.current
        dwell = int(t_ms - self.last_switch_ms)
        switched = False

        if self.current is None:
            self.current, self.last_switch_ms, switched = challenger, t_ms, True
        elif challenger != self.current:
            lead = sm[EMOTIONS.index(challenger)] - sm[EMOTIONS.index(self.current)]
            if lead >= self.margin and dwell >= self.min_dwell_ms:
                self.current, self.last_switch_ms, switched = challenger, t_ms, True
            else:
                self.suppressed += 1

        return BeliefOut(sm, self.current, raw, prev, switched, dwell, self.suppressed)

    @property
    def description(self) -> str:
        return (f"ema(a={self.alpha})+hysteresis(m={self.margin})"
                f"+dwell({self.min_dwell_ms}ms)")

"""Emotional inertia, measured two ways.

F1 on its own is the wrong metric here. Smoothing raises F1 on MELD partly
because MELD emotions are sticky inside a dialogue, and that's a property of the
dataset rather than evidence the lamp looks less broken. So I report both:

  * weighted F1, raw vs smoothed. Did I lose accuracy?
  * switch rate, switches per minute of speech. Did the lamp stop flickering?

The second number is the one the behaviour layer actually pays for.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from ..config import EMA_ALPHA, EMOTIONS, HYSTERESIS_MARGIN, MIN_DWELL_MS
from ..data.features import SplitFeatures
from ..runtime.belief import BeliefTracker
from .metrics import load_head, predict_probs


def _dialogue_order(fx: SplitFeatures):
    order = np.lexsort((fx.utterance_id, fx.dialogue_id))
    groups, cur, prev = [], [], None
    for i in order:
        d = fx.dialogue_id[i]
        if prev is not None and d != prev:
            groups.append(cur)
            cur = []
        cur.append(int(i))
        prev = d
    if cur:
        groups.append(cur)
    return groups


def _sequential(probs, fx, groups, alpha, margin, dwell):
    """Run one smoothing configuration over every dialogue; return (wF1, switches)."""
    pred = np.zeros(len(fx), dtype=int)
    switches, minutes = 0, 0.0
    for g in groups:
        tracker = BeliefTracker(alpha=alpha, margin=margin, min_dwell_ms=dwell)
        t_ms, last = 0.0, None
        for i in g:
            out = tracker.update(probs[i], t_ms)
            pred[i] = EMOTIONS.index(out.emotion)
            if last is not None and out.emotion != last:
                switches += 1
            last = out.emotion
            t_ms += float(fx.duration_s[i]) * 1000.0
        minutes += t_ms / 60000.0
    return (round(float(f1_score(fx.labels, pred, average="weighted",
                                 zero_division=0)), 4),
            switches, round(switches / max(minutes, 1e-6), 2))


def sweep(fx: SplitFeatures, head: str = "fused") -> list[dict]:
    """The accuracy/stability frontier.

    Smoothing is a product decision, not an accuracy optimisation: the right
    operating point depends on how twitchy a lamp is allowed to look, which is a
    question for a human study rather than for weighted-F1. So we publish the
    frontier and mark the point we shipped instead of quoting a single number.
    """
    m, ck = load_head(head)
    probs = predict_probs(m, ck, fx)
    groups = _dialogue_order(fx)
    rows = []
    for alpha in (1.0, 0.8, 0.6, 0.4):
        for margin in (0.0, 0.10, 0.15, 0.25):
            f1, sw, spm = _sequential(probs, fx, groups, alpha, margin,
                                      MIN_DWELL_MS)
            rows.append({"alpha": alpha, "margin": margin,
                         "min_dwell_ms": MIN_DWELL_MS, "weighted_f1": f1,
                         "switches": sw, "switches_per_minute": spm,
                         "shipped": alpha == EMA_ALPHA and margin == HYSTERESIS_MARGIN})
    return rows


def run(fx: SplitFeatures, head: str = "fused", trace_dialogue: int | None = None) -> dict:
    m, ck = load_head(head)
    probs = predict_probs(m, ck, fx)
    groups = _dialogue_order(fx)

    raw_pred = np.zeros(len(fx), dtype=int)
    sm_pred = np.zeros(len(fx), dtype=int)
    raw_sw = sm_sw = 0
    total_minutes = 0.0
    traces: dict[int, list] = {}

    for g in groups:
        tracker = BeliefTracker()
        t_ms = 0.0
        last_raw = last_sm = None
        trace = []
        for i in g:
            out = tracker.update(probs[i], t_ms)
            r = EMOTIONS[int(probs[i].argmax())]
            raw_pred[i] = EMOTIONS.index(r)
            sm_pred[i] = EMOTIONS.index(out.emotion)
            if last_raw is not None and r != last_raw:
                raw_sw += 1
            if last_sm is not None and out.emotion != last_sm:
                sm_sw += 1
            last_raw, last_sm = r, out.emotion
            trace.append({
                "uid": str(fx.uids[i]), "t_ms": round(t_ms, 1),
                "text": str(fx.texts[i])[:90],
                "gold": EMOTIONS[int(fx.labels[i])],
                "raw": r, "smoothed": out.emotion,
                "confidence": round(float(out.probs.max()), 4),
            })
            t_ms += float(fx.duration_s[i]) * 1000.0
        total_minutes += t_ms / 60000.0
        traces[int(fx.dialogue_id[g[0]])] = trace

    def sc(p):
        return {
            "weighted_f1": round(float(f1_score(fx.labels, p, average="weighted",
                                                zero_division=0)), 4),
            "macro_f1": round(float(f1_score(fx.labels, p, average="macro",
                                             zero_division=0)), 4),
            "accuracy": round(float((fx.labels == p).mean()), 4),
        }

    out = {
        "smoothing": BeliefTracker().description,
        "n_dialogues": len(groups),
        "total_speech_minutes": round(total_minutes, 2),
        "raw": {**sc(raw_pred), "switches": raw_sw,
                "switches_per_minute": round(raw_sw / max(total_minutes, 1e-6), 2)},
        "smoothed": {**sc(sm_pred), "switches": sm_sw,
                     "switches_per_minute": round(sm_sw / max(total_minutes, 1e-6), 2)},
        "note": ("Total speech minutes sums utterance durations only; MELD does "
                 "not give inter-turn gaps, so the real-world switch rate would "
                 "be lower still."),
    }
    out["frontier"] = sweep(fx, head)
    out["switch_reduction_pct"] = round(
        100 * (1 - sm_sw / max(raw_sw, 1)), 1)
    out["weighted_f1_delta"] = round(
        out["smoothed"]["weighted_f1"] - out["raw"]["weighted_f1"], 4)
    if trace_dialogue is not None and trace_dialogue in traces:
        out["trace"] = {"dialogue_id": trace_dialogue,
                        "turns": traces[trace_dialogue]}
    else:
        # Pick the dialogue where smoothing changed the most decisions -- the
        # honest choice is the most illustrative one, not the prettiest one.
        best = max(traces.items(),
                   key=lambda kv: sum(t["raw"] != t["smoothed"] for t in kv[1]))
        out["trace"] = {"dialogue_id": best[0], "turns": best[1]}
    return out

"""Is the vision channel load-bearing, or decoration?

A single hand-picked "look, it changed!" example proves nothing -- with 2,610
test utterances you can always find one. So we measure the swap in aggregate:
hold the transcript fixed, substitute a DIFFERENT utterance's video, and look at
how far the posterior moves across the whole test set. Then we do the mirror
experiment (fix the video, swap the text) to size text dominance honestly.

If the vision swap moves nothing, the correct conclusion is that our fusion is
decoration, and we say so.
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import EMOTIONS
from ..data.features import SplitFeatures
from ..runtime.policy import decide, project_va
from .metrics import load_head


def _probs(m, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return m.probs(torch.from_numpy(t), torch.from_numpy(v)).numpy()


def run(fx: SplitFeatures, seed: int = 0, n_examples: int = 6) -> dict:
    m, ck = load_head("fused")
    off = np.array(ck["va_offset"])
    # Build the text input from the fields THIS head was trained on, rather
    # than assuming one. Hardcoding `text_ctx` here silently broke the moment
    # the fused head started consuming two text encodings.
    text = np.concatenate([fx.__dict__[f] for f in ck["text_fields"]], 1)
    vis = fx.vision
    n = len(fx)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    base = _probs(m, text, vis)
    swap_v = _probs(m, text, vis[perm])      # same words, someone else's video
    swap_t = _probs(m, text[perm], vis)      # same video, someone else's words

    def tv(a, b):
        return 0.5 * np.abs(a - b).sum(1)

    def intents(p):
        out = []
        for row in p:
            v, a = project_va(row, off)
            out.append(decide(row, float(row.max()), v, a).intent)
        return np.array(out)

    i_base, i_sv, i_st = intents(base), intents(swap_v), intents(swap_t)
    tv_v, tv_t = tv(base, swap_v), tv(base, swap_t)

    summary = {
        "n": int(n),
        "vision_swap": {
            "mean_tv": round(float(tv_v.mean()), 4),
            "median_tv": round(float(np.median(tv_v)), 4),
            "p90_tv": round(float(np.quantile(tv_v, 0.9)), 4),
            "emotion_flip_rate": round(float((base.argmax(1) != swap_v.argmax(1)).mean()), 4),
            "behaviour_intent_change_rate": round(float((i_base != i_sv).mean()), 4),
        },
        "text_swap": {
            "mean_tv": round(float(tv_t.mean()), 4),
            "median_tv": round(float(np.median(tv_t)), 4),
            "p90_tv": round(float(np.quantile(tv_t, 0.9)), 4),
            "emotion_flip_rate": round(float((base.argmax(1) != swap_t.argmax(1)).mean()), 4),
            "behaviour_intent_change_rate": round(float((i_base != i_st).mean()), 4),
        },
        "interpretation": (
            "Vision-swap TV >> 0 means the visual stream genuinely moves the "
            "posterior. Compare the two blocks to size text dominance: if "
            "text_swap dwarfs vision_swap, the model is mostly reading words, "
            "which is the expected and honest finding on MELD."
        ),
    }

    # Concrete pairs for the demo: same transcript, different video, DIFFERENT
    # behaviour intent. Ranked by posterior movement so the examples are the
    # strongest real ones, not cherry-picked anecdotes.
    changed = np.where(i_base != i_sv)[0]
    changed = changed[np.argsort(-tv_v[changed])][:n_examples]
    summary["examples"] = [{
        "uid": str(fx.uids[i]),
        "text": str(fx.texts[i])[:110],
        "gold": EMOTIONS[int(fx.labels[i])],
        "with_own_video": {
            "emotion": EMOTIONS[int(base[i].argmax())],
            "confidence": round(float(base[i].max()), 4),
            "intent": str(i_base[i]),
        },
        "with_video_from": str(fx.uids[perm[i]]),
        "with_swapped_video": {
            "emotion": EMOTIONS[int(swap_v[i].argmax())],
            "confidence": round(float(swap_v[i].max()), 4),
            "intent": str(i_sv[i]),
        },
        "total_variation": round(float(tv_v[i]), 4),
    } for i in changed]
    return summary

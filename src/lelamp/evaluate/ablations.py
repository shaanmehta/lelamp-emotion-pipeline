"""Modality ablation, per-class breakdown, and two honesty checks: a leak hunt
and the text-ambiguous subset.

MELD is known to be text-dominant. The interesting question for a robot isn't
"does fusion raise weighted F1", because it barely will. It's "where does vision
buy anything, and is that somewhere the behaviour layer cares about".
"""
from __future__ import annotations

import numpy as np

from ..config import EMOTIONS
from ..data.features import SplitFeatures
from .metrics import (confusion, ece, load_head, majority_baseline, per_class_f1,
                      predict_probs, scores)

CONFIGS = ["text_solo", "text_ctx", "text_pair", "vision",
           "fused_ctx_only", "fused_concat", "fused"]

# MELD's six principal cast appear in train, dev AND test. A vision head can get
# credit for recognising an actor rather than an expression, so we check the gap.
MAIN_CAST = {"Chandler", "Joey", "Monica", "Phoebe", "Rachel", "Ross"}


def run(feats: dict[str, SplitFeatures], train_counts: np.ndarray) -> dict:
    test = feats["test"]
    out: dict = {"n_test": len(test),
                 "test_class_counts": {e: int(c) for e, c in
                                       zip(EMOTIONS, test.class_counts)},
                 "majority": majority_baseline(train_counts, test),
                 "configs": {}}

    probs: dict[str, np.ndarray] = {}
    for name in CONFIGS:
        m, ck = load_head(name)
        p = predict_probs(m, ck, test)
        probs[name] = p
        pred = p.argmax(1)
        e, _ = ece(p, test.labels)
        out["configs"][name] = {
            **scores(test.labels, pred),
            "per_class_f1": per_class_f1(test.labels, pred),
            "ece": round(e, 4),
            "temperature": round(float(ck["temperature"]), 4),
            "head_params": int(sum(v.numel() for v in ck["state_dict"].values())) + 1,
        }

    # --- where does vision actually help? --------------------------------
    # Compare fusion against the STRONGEST text-only config, not a convenient
    # one. text_ctx is weaker than text_solo (mean-pooling dilutes the current
    # utterance), and measuring fusion against the weaker baseline would flatter
    # it for no reason.
    best_text = max(("text_solo", "text_ctx", "text_pair"),
                    key=lambda n: out["configs"][n]["weighted_f1"])
    out["best_text_only_config"] = best_text
    t_probs, f_probs = probs[best_text], probs["fused"]
    margin = np.sort(t_probs, 1)[:, -1] - np.sort(t_probs, 1)[:, -2]
    amb = margin < np.quantile(margin, 0.30)     # text least sure: bottom 30%
    out["text_ambiguous_subset"] = {
        "definition": f"bottom-30% top-2 margin of the best text-only config "
                      f"({best_text})",
        "n": int(amb.sum()),
        "text_only": scores(test.labels[amb], t_probs[amb].argmax(1)),
        "fused": scores(test.labels[amb], f_probs[amb].argmax(1)),
    }
    out["per_class_delta_fused_minus_text"] = {
        e: round(out["configs"]["fused"]["per_class_f1"][e]
                 - out["configs"][best_text]["per_class_f1"][e], 4)
        for e in EMOTIONS
    }
    out["confusion_fused"] = confusion(test.labels, f_probs.argmax(1)).tolist()
    out["confusion_text_only"] = confusion(test.labels, t_probs.argmax(1)).tolist()

    # The pair a lamp's behaviour depends on most: neutral vs anger is the
    # difference between "idle" and "recoil".
    def pair_acc(p, a, b) -> float:
        """Accuracy restricted to utterances whose gold label is a or b.

        These are the distinctions the behaviour layer actually depends on:
        neutral vs anger is the difference between `idle` and `recoil`.
        """
        ia, ib = EMOTIONS.index(a), EMOTIONS.index(b)
        m = (test.labels == ia) | (test.labels == ib)
        if not m.any():
            return 0.0
        return round(float((p[m].argmax(1) == test.labels[m]).mean()), 4)

    out["behaviour_critical_pairs"] = {
        f"{a}_vs_{b}": {"text_only": pair_acc(t_probs, a, b),
                        "fused": pair_acc(f_probs, a, b),
                        "delta": round(pair_acc(f_probs, a, b)
                                       - pair_acc(t_probs, a, b), 4)}
        for a, b in [("neutral", "anger"), ("neutral", "surprise"),
                     ("neutral", "joy"), ("neutral", "sadness")]
    }

    # --- leak hunt --------------------------------------------------------
    is_main = np.array([s in MAIN_CAST for s in test.speakers])
    out["leak_check"] = {
        "note": ("MELD's six principal cast are present in every split, so a "
                 "vision head can be rewarded for recognising an actor rather "
                 "than an expression. A large main-cast/other gap on the "
                 "vision-only config is the tell."),
        "main_cast_frac": round(float(is_main.mean()), 4),
        **{name: {"main_cast": scores(test.labels[is_main], probs[name][is_main].argmax(1)),
                  "other": scores(test.labels[~is_main], probs[name][~is_main].argmax(1))}
           for name in CONFIGS},
        "dialogue_id_note": (
            "MELD numbers dialogues from 0 WITHIN each split, so raw id overlap "
            "across splits is expected and meaningless -- the splits are "
            "disjoint by construction (different episodes). We verified this "
            "matters for clip vendoring too: dia237_utt3 exists in both train "
            "and test and they are different clips."),
    }
    # --- does the fused head survive losing the camera? -------------------
    # A lamp's camera is useless far more often than a benchmark suggests: user
    # out of frame, back turned, dark room. The fused head is trained with
    # modality dropout specifically so a working text-only path exists inside
    # it. This measures whether that worked, by zeroing the vision features at
    # inference and re-scoring.
    m_f, ck_f = load_head("fused")
    import torch
    t_in = np.concatenate([test.__dict__[f] for f in ck_f["text_fields"]], 1)
    with torch.no_grad():
        p_blind = m_f.probs(torch.from_numpy(t_in),
                            torch.zeros(len(test), test.vision.shape[1]),
                            calibrated=True).numpy()
    out["vision_dropout_robustness"] = {
        "note": ("fused head evaluated with the vision features zeroed, i.e. the "
                 "camera is unavailable at inference time"),
        "fused_with_vision": scores(test.labels, f_probs.argmax(1)),
        "fused_vision_zeroed": scores(test.labels, p_blind.argmax(1)),
        "best_text_only": scores(test.labels, t_probs.argmax(1)),
    }

    out["vision_coverage"] = {
        s: round(float(feats[s].has_vision.mean()), 4) for s in feats
    }
    return out

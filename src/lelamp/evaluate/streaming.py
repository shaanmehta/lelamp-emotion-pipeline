"""Partial-utterance behaviour: the accuracy-vs-coverage curve, and a learned
commit-vs-wait policy.

This is the plot that matters most for a robot and that almost nobody produces.
MELD hands you whole utterances; a lamp in a kitchen gets words one at a time and
has to decide, continuously, whether it has heard enough to react.

Two axes, because they answer different questions:
  * vs. fraction heard -- how much of a sentence carries the emotion?
  * vs. elapsed milliseconds -- what the robot's clock actually sees.

IMPORTANT CAVEAT, stated here because it changes how the curve should be read:
the fusion head is trained on WHOLE utterances and evaluated here on prefixes.
So this measures how gracefully a full-utterance model degrades on partial
input, not how well a model trained for streaming would do. The curve is
therefore a LOWER bound -- prefix-augmented training would raise the early part
of it, and that is a listed follow-up rather than something claimed here.

The commit policy is learned by fitted-Q backward induction on dev and reported
on test. Being precise about the setup: we observe the reward for EVERY prefix,
not just the one we chose, so this is offline policy learning with
full-information feedback -- strictly easier than the online bandit problem a
deployed robot faces. Calling it a contextual bandit would overstate it.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score

from ..config import FEATURES
from ..data.features import SplitFeatures
from ..models.heads import pool_frames
from .metrics import load_head


def _prefix_probs(split: str, fx: SplitFeatures, head: str = "fused"):
    """Fused posteriors at every text-prefix / frame-prefix level.

    Vision is truncated in lockstep with text: at fraction f the lamp has seen
    ceil(f * T) frames, because frames and words arrive together in real time.
    """
    z = np.load(FEATURES / f"text_{split}_prefix.npz", allow_pickle=False)
    fracs, labels = z["fracs"], z["labels"]
    order = {str(u): i for i, u in enumerate(z["uids"])}
    idx = np.array([order[u] for u in fx.uids])

    m, ck = load_head(head)
    # Assemble exactly the text fields this head was trained on, so the curve is
    # computed with the same model that the ablation table reports.
    key = {"text_solo": "solo", "text_ctx": "ctx"}
    fields = ck["text_fields"] if ck["use_text"] else []
    for f_ in fields:
        if key[f_] not in z:
            raise FileNotFoundError(
                f"prefix cache lacks '{key[f_]}' features for split '{split}'. "
                f"Run `python scripts/extract_solo_prefix.py`.")

    out = np.zeros((len(fracs), len(fx), 7), dtype=np.float32)
    for fi, f in enumerate(fracs):
        vis = np.stack([
            pool_frames(fr[: max(1, int(np.ceil(f * len(fr))))] if len(fr) else fr)
            for fr in fx.frames
        ]).astype(np.float32)
        txt = (np.concatenate([z[key[f_]][fi][idx] for f_ in fields], 1)
               .astype(np.float32) if fields else None)
        with torch.no_grad():
            p = m.probs(None if txt is None else torch.from_numpy(txt),
                        torch.from_numpy(vis), calibrated=True)
        out[fi] = p.numpy()
    return fracs, out, labels[idx]


def coverage_curve(fx: SplitFeatures, split: str = "test") -> dict:
    fracs, probs, y = _prefix_probs(split, fx)
    dur_ms = fx.duration_s * 1000.0
    rows = []
    for fi, f in enumerate(fracs):
        pred = probs[fi].argmax(1)
        rows.append({
            "frac": round(float(f), 2),
            "mean_elapsed_ms": round(float((dur_ms * f).mean()), 1),
            "weighted_f1": round(float(f1_score(y, pred, average="weighted",
                                                zero_division=0)), 4),
            "accuracy": round(float((pred == y).mean()), 4),
            "mean_confidence": round(float(probs[fi].max(1).mean()), 4),
        })
    full = rows[-1]["weighted_f1"]
    # The operationally interesting number: the earliest point at which we are
    # already within one F1 point of what the whole utterance would give.
    reach = next((r["frac"] for r in rows if r["weighted_f1"] >= full - 0.01), 1.0)
    return {"curve": rows, "full_weighted_f1": full,
            "frac_to_within_1pt": reach,
            "mean_utterance_ms": round(float(dur_ms.mean()), 1)}


# --------------------------------------------------------------------------
# commit-vs-wait
# --------------------------------------------------------------------------
def _context(probs_k: np.ndarray, k: int, n_steps: int) -> np.ndarray:
    """Features available to the policy at decision step k. All causal."""
    s = np.sort(probs_k, 1)
    conf = s[:, -1]
    margin = s[:, -1] - s[:, -2]
    ent = -(probs_k * np.log(probs_k + 1e-9)).sum(1) / np.log(probs_k.shape[1])
    step = np.full_like(conf, k / (n_steps - 1))
    return np.stack([np.ones_like(conf), conf, margin, ent, step], 1)


def _ridge(X: np.ndarray, y: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def _fit_policy(P_d, y_d, K, lam_delay):
    """Fitted-Q backward induction. Returns per-step (w_commit, w_wait).

    Both arms are fitted by the SAME ridge regression on the SAME features. The
    first version of this estimated the commit arm with raw calibrated
    confidence and only the wait arm by regression; that asymmetry systematically
    over-valued waiting and produced a policy that hardly ever committed and lost
    badly to a fixed threshold. Estimator symmetry is doing real work here.
    """
    correct = (P_d.argmax(2) == y_d[None, :]).astype(np.float64)
    delay = (np.arange(K) / (K - 1))[:, None] * lam_delay
    R = correct - delay                                  # reward of committing at k

    V = R[-1].copy()                                     # last step: forced commit
    W: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in range(K - 2, -1, -1):
        X = _context(P_d[k], k, K)
        w_commit = _ridge(X, R[k])
        w_wait = _ridge(X, V)
        W[k] = (w_commit, w_wait)
        V = np.maximum(X @ w_commit, X @ w_wait)
    return W


def _rollout(P, y, K, lam_delay, decide_fn, fracs=None):
    n = P.shape[1]
    step = np.full(n, K - 1)
    done = np.zeros(n, bool)
    for k in range(K - 1):
        act = decide_fn(P[k], k) & ~done
        step[act] = k
        done |= act
    pred = P[step, np.arange(n)].argmax(1)
    # Step 0 is the FIRST decision point, which is 10% of the utterance heard --
    # not 0%. Report the real coverage so the table cannot be misread.
    heard = np.asarray(fracs)[step] if fracs is not None else step / (K - 1)
    frac = step / (K - 1)                       # normalised delay for the reward
    return {
        "mean_heard": round(float(np.mean(heard)), 4),
        "weighted_f1": round(float(f1_score(y, pred, average="weighted",
                                            zero_division=0)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "mean_frac_heard": round(float(frac.mean()), 4),
        "mean_reward": round(float(((pred == y) - lam_delay * frac).mean()), 4),
    }


def learn_commit_policy(dev: SplitFeatures, test: SplitFeatures,
                        lam_delay: float = 0.35,
                        lam_sweep: tuple = (0.15, 0.25, 0.35, 0.50)) -> dict:
    """Learn when to stop waiting. Fit on dev, report on test.

    Reward for committing at step k: 1 if correct, minus lam_delay * k/(K-1).
    lam_delay is the exchange rate between accuracy and latency -- a PRODUCT
    decision, not a learned one -- so we sweep it rather than quoting one value.

    Being precise about the setup: we observe the reward for EVERY prefix, not
    only the one we chose, so this is offline policy learning with
    full-information feedback. That is strictly easier than the online bandit a
    deployed robot faces, and calling it a contextual bandit would overstate it.
    """
    fr_d, P_d, y_d = _prefix_probs("dev", dev)
    fr_t, P_t, y_t = _prefix_probs("test", test)
    K = len(fr_d)

    def evaluate(lam: float) -> dict:
        W = _fit_policy(P_d, y_d, K, lam)

        def commit_now(p: np.ndarray, k: int) -> np.ndarray:
            if k not in W:
                return np.ones(len(p), bool)
            X = _context(p, k, K)
            w_c, w_w = W[k]
            return (X @ w_c) >= (X @ w_w)

        learned = _rollout(P_t, y_t, K, lam, commit_now, fr_t)
        base = {}
        for c in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
            base[f"fixed_conf_{c:.2f}"] = _rollout(
                P_t, y_t, K, lam, lambda p, k, c=c: p.max(1) >= c, fr_t)
        for k0 in (1, 2, 4, 6, 8):
            base[f"fixed_step_{k0}"] = _rollout(
                P_t, y_t, K, lam, lambda p, k, k0=k0: np.full(len(p), k >= k0), fr_t)
        base["always_wait"] = _rollout(
            P_t, y_t, K, lam, lambda p, k: np.zeros(len(p), bool), fr_t)
        best = max(base.items(), key=lambda kv: kv[1]["mean_reward"])
        return {
            "lambda_delay": lam,
            "learned_policy": learned,
            "baselines": base,
            "best_fixed_baseline": {"name": best[0], **best[1]},
            "reward_gain_vs_best_fixed": round(
                learned["mean_reward"] - best[1]["mean_reward"], 4),
        }

    main = evaluate(lam_delay)
    main["setup"] = ("offline fitted-Q backward induction on dev, "
                     "full-information rewards (every prefix is evaluable); "
                     "reported on MELD test")
    main["sweep"] = [
        {"lambda_delay": lam,
         "learned_reward": r["learned_policy"]["mean_reward"],
         "learned_frac_heard": r["learned_policy"]["mean_heard"],
         "learned_wf1": r["learned_policy"]["weighted_f1"],
         "best_fixed": r["best_fixed_baseline"]["name"],
         "best_fixed_reward": r["best_fixed_baseline"]["mean_reward"],
         "gain": r["reward_gain_vs_best_fixed"]}
        for lam in lam_sweep for r in [evaluate(lam)]
    ]
    return main

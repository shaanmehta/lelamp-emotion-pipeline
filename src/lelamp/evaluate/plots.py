"""Every figure in the README, regenerated from artifacts/results.json."""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import EMOTIONS, FIGURES  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, name):
    p = FIGURES / name
    # Figures with a twin axis or a shared colourbar are not tight_layout
    # compatible; bbox_inches="tight" already handles those.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def coverage(res: dict):
    """THE plot: how much of an utterance do you need before you can react?"""
    c = res["streaming"]["coverage_curve"]["curve"]
    fr = [r["frac"] for r in c]
    f1 = [r["weighted_f1"] for r in c]
    ms = [r["mean_elapsed_ms"] for r in c]
    conf = [r["mean_confidence"] for r in c]
    full = res["streaming"]["coverage_curve"]["full_weighted_f1"]

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax[0].plot(fr, f1, "o-", lw=2, color="#2b6cb0", label="fused weighted-F1")
    ax[0].axhline(full, ls="--", lw=1, color="#888", label="full utterance")
    ax[0].axhline(full - 0.01, ls=":", lw=1, color="#bbb")
    ax[0].set_xlabel("fraction of utterance heard")
    ax[0].set_ylabel("weighted F1 (MELD test)")
    ax[0].set_title("Accuracy vs. coverage")
    ax[0].legend(fontsize=8, loc="lower right")

    ax[1].plot(ms, f1, "o-", lw=2, color="#2b6cb0")
    ax[1].set_xlabel("mean elapsed time since utterance onset (ms)")
    ax[1].set_ylabel("weighted F1")
    ax[1].set_title("Same curve on the robot's clock")
    a2 = ax[1].twinx()
    a2.plot(ms, conf, "s--", lw=1, ms=3, color="#c05621", alpha=.8)
    a2.set_ylabel("mean confidence", color="#c05621")
    a2.grid(False)
    return _save(fig, "01_coverage_curve.png")


def reliability(res: dict):
    bins = res["calibration"]["bins_calibrated"]
    raw = res["calibration"]["bins_uncalibrated"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, b, title, e in ((ax[0], raw, "Before temperature scaling",
                            res["calibration"]["ece_uncalibrated"]),
                           (ax[1], bins, "After temperature scaling",
                            res["calibration"]["ece_calibrated"])):
        xs = [(x["lo"] + x["hi"]) / 2 for x in b if x["n"] > 0]
        ys = [x["acc"] for x in b if x["n"] > 0]
        ns = [x["n"] for x in b if x["n"] > 0]
        a.plot([0, 1], [0, 1], "--", color="#aaa", lw=1)
        a.bar(xs, ys, width=0.055, color="#2b6cb0", alpha=.75,
              edgecolor="white")
        a.scatter(xs, ys, s=[max(6, n / 12) for n in ns], color="#c05621", zorder=3)
        a.set_xlim(0, 1); a.set_ylim(0, 1)
        a.set_xlabel("confidence"); a.set_ylabel("accuracy")
        a.set_title(f"{title}  (ECE={e:.3f})")
    return _save(fig, "02_reliability.png")


def risk_coverage(res: dict):
    rc = res["calibration"]["risk_coverage"]
    cov = [r["coverage"] for r in rc]
    f1 = [r["weighted_f1"] for r in rc]
    tau = res["calibration"]["tau_abstain"]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(cov, f1, "-", lw=2, color="#2b6cb0")
    pick = min(rc, key=lambda r: abs(r["tau"] - tau))
    ax.scatter([pick["coverage"]], [pick["weighted_f1"]], s=60, zorder=4,
               color="#c05621",
               label=f"shipped $\\tau$={tau}\ncoverage={pick['coverage']:.2f}")
    ax.set_xlabel("coverage (fraction of utterances the lamp commits on)")
    ax.set_ylabel("weighted F1 on committed utterances")
    ax.set_title("Abstention buys accuracy")
    ax.legend(fontsize=8)
    return _save(fig, "03_risk_coverage.png")


def smoothing(res: dict):
    s = res["smoothing"]
    tr = s["trace"]["turns"]
    idx = np.arange(len(tr))
    gold = [EMOTIONS.index(t["gold"]) for t in tr]
    raw = [EMOTIONS.index(t["raw"]) for t in tr]
    sm = [EMOTIONS.index(t["smoothed"]) for t in tr]

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6),
                           gridspec_kw={"width_ratios": [2.1, 1]})
    ax[0].step(idx, gold, where="mid", lw=3, alpha=.30, color="#333", label="gold")
    ax[0].step(idx, raw, where="mid", lw=1.4, ls="--", color="#c05621",
               label="raw argmax")
    ax[0].step(idx, sm, where="mid", lw=2, color="#2b6cb0", label="smoothed")
    ax[0].set_yticks(range(len(EMOTIONS)))
    ax[0].set_yticklabels(EMOTIONS)
    ax[0].set_xlabel(f"turn index (test dialogue {s['trace']['dialogue_id']})")
    ax[0].set_title("Emotional inertia on a real dialogue")
    ax[0].legend(fontsize=8, ncol=3, loc="upper left")

    labels = ["raw", "smoothed"]
    vals = [s["raw"]["switches_per_minute"], s["smoothed"]["switches_per_minute"]]
    f1s = [s["raw"]["weighted_f1"], s["smoothed"]["weighted_f1"]]
    b = ax[1].bar(labels, vals, color=["#c05621", "#2b6cb0"], alpha=.85, width=.55)
    for r, v, f in zip(b, vals, f1s):
        ax[1].text(r.get_x() + r.get_width() / 2, v, f"{v:.1f}\nwF1 {f:.3f}",
                   ha="center", va="bottom", fontsize=8)
    ax[1].set_ylabel("emotion switches per minute of speech")
    ax[1].set_ylim(0, max(vals) * 1.35)
    ax[1].set_title("What the behaviour layer pays")
    return _save(fig, "04_smoothing.png")


def confusion(res: dict):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for a, key, title in ((ax[0], "confusion_text_only",
                           f"best text-only ({res['ablations']['best_text_only_config']})"),
                          (ax[1], "confusion_fused", "fused (text+vision)")):
        cm = np.array(res["ablations"][key], dtype=float)
        cmn = cm / cm.sum(1, keepdims=True).clip(1)
        im = a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        a.set_xticks(range(len(EMOTIONS)))
        a.set_xticklabels(EMOTIONS, rotation=45, ha="right")
        a.set_yticks(range(len(EMOTIONS)))
        a.set_yticklabels(EMOTIONS)
        a.set_title(f"{title}  (row-normalised)")
        a.grid(False)
        for i in range(len(EMOTIONS)):
            for j in range(len(EMOTIONS)):
                if cmn[i, j] > 0.02:
                    a.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                           fontsize=7,
                           color="white" if cmn[i, j] > 0.5 else "#333")
        a.set_ylabel("gold"); a.set_xlabel("predicted")
    fig.colorbar(im, ax=ax, fraction=0.02)
    return _save(fig, "05_confusion.png")


def ablation_bars(res: dict):
    ab = res["ablations"]
    names = ["majority", "vision", "text_ctx", "text_solo", "text_pair",
             "fused_ctx_only", "fused"]
    vals = [ab["majority"]["weighted_f1"]] + [
        ab["configs"][n]["weighted_f1"] for n in names[1:]]
    macro = [ab["majority"]["macro_f1"]] + [
        ab["configs"][n]["macro_f1"] for n in names[1:]]

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6),
                           gridspec_kw={"width_ratios": [1.1, 1.3]})
    x = np.arange(len(names))
    ax[0].bar(x - .2, vals, .38, label="weighted F1", color="#2b6cb0")
    ax[0].bar(x + .2, macro, .38, label="macro F1", color="#c05621", alpha=.85)
    ax[0].axhspan(0.64, 0.67, color="#38a169", alpha=.12)
    ax[0].text(0.05, 0.655, "published MELD range", fontsize=7, color="#276749")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=30, ha="right",
                                               fontsize=7.5)
    ax[0].set_ylabel("F1 (MELD test)"); ax[0].legend(fontsize=8)
    ax[0].set_title("Baselines and modality ablation")

    delta = res["ablations"]["per_class_delta_fused_minus_text"]
    ks = sorted(delta, key=lambda k: delta[k])
    ax[1].barh(ks, [delta[k] for k in ks],
               color=["#c05621" if delta[k] < 0 else "#2b6cb0" for k in ks])
    ax[1].axvline(0, color="#333", lw=1)
    ax[1].set_xlabel("per-class F1: fused − text-only")
    ax[1].set_title("Where vision actually changes something")
    return _save(fig, "06_ablation.png")


def counterfactual(res: dict):
    cf = res["counterfactual"]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    labels = ["swap the video\n(same words)", "swap the words\n(same video)"]
    vals = [cf["vision_swap"]["mean_tv"], cf["text_swap"]["mean_tv"]]
    flips = [cf["vision_swap"]["behaviour_intent_change_rate"],
             cf["text_swap"]["behaviour_intent_change_rate"]]
    b = ax.bar(labels, vals, color=["#2b6cb0", "#c05621"], alpha=.85, width=.55)
    for r, v, f in zip(b, vals, flips):
        ax.text(r.get_x() + r.get_width() / 2, v,
                f"TV {v:.3f}\nintent changes {f*100:.0f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("mean total-variation shift in posterior")
    ax.set_ylim(0, max(vals) * 1.4)
    ax.set_title("Counterfactual: is fusion load-bearing?\n(all 2,610 test utterances)")
    return _save(fig, "07_counterfactual.png")


def latency(res: dict):
    lat = res["latency"]["stages"]
    order = [k for k in ("decode_mp4_ms", "clip_1frame_ms", "reflex_total_ms",
                         "roberta_ms", "clip_8frame_ms", "fusion_heads_ms",
                         "belief_ms", "policy_ms", "deliberative_total_ms",
                         "ttft_ms") if k in lat]
    p50 = [lat[k]["p50"] for k in order]
    p95 = [lat[k]["p95"] for k in order]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.barh(y - .19, p50, .36, label="p50", color="#2b6cb0")
    ax.barh(y + .19, p95, .36, label="p95", color="#c05621", alpha=.85)
    ax.set_yticks(y)
    ax.set_yticklabels([k.replace("_ms", "") for k in order])
    ax.set_xscale("log")
    ax.set_xlabel("milliseconds (log scale)")
    ax.axvline(res["latency"]["budgets"]["reflexive_ms"], ls="--", color="#38a169",
               lw=1.2, label="reflexive budget 300 ms")
    ax.axvline(res["latency"]["budgets"]["ttft_ms"], ls=":", color="#805ad5",
               lw=1.2, label="TTFT budget 800 ms")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title(f"Measured latency — {res['latency']['hardware']['device']}, "
                 f"{res['latency']['responder']}")
    ax.invert_yaxis()
    return _save(fig, "08_latency.png")


def all_figures(res: dict) -> list:
    out = []
    for fn in (coverage, reliability, risk_coverage, smoothing, confusion,
               ablation_bars, counterfactual, latency):
        try:
            out.append(str(fn(res)))
        except Exception as e:  # a missing section must not kill the rest
            print(f"  [plots] skipped {fn.__name__}: {type(e).__name__}: {e}")
    return out

"""Regenerates every number and figure in the README. One command.

    python -m lelamp.evaluate.run_all

Writes artifacts/results.json and artifacts/figures/*.png, then prints the
summary table the README quotes.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from ..config import ARTIFACTS, EMOTIONS, TAU_ABSTAIN
from ..data.features import load_all
from . import ablations, counterfactual, plots, smoothing, streaming
from .metrics import ece, load_head, predict_probs, risk_coverage


def calibration_section(feats) -> dict:
    test = feats["test"]
    m, ck = load_head("fused")
    p_cal = predict_probs(m, ck, test, calibrated=True)
    p_raw = predict_probs(m, ck, test, calibrated=False)
    e_cal, bins_cal = ece(p_cal, test.labels)
    e_raw, bins_raw = ece(p_raw, test.labels)
    rc = risk_coverage(p_cal, test.labels)
    pick = min(rc, key=lambda r: abs(r["tau"] - TAU_ABSTAIN))
    return {
        "temperature": round(float(ck["temperature"]), 4),
        "ece_uncalibrated": round(e_raw, 4),
        "ece_calibrated": round(e_cal, 4),
        "bins_uncalibrated": bins_raw,
        "bins_calibrated": bins_cal,
        "risk_coverage": rc,
        "tau_abstain": TAU_ABSTAIN,
        "at_shipped_tau": pick,
        "abstain_rate_at_shipped_tau": round(1 - pick["coverage"], 4),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-latency", action="store_true",
                    help="skip the latency benchmark (needs clips + models)")
    ap.add_argument("--no-llm", action="store_true",
                    help="latency benchmark uses the template responder only")
    ap.add_argument("--lambda-delay", type=float, default=0.35)
    a = ap.parse_args(argv)

    t0 = time.time()
    feats = load_all()
    res: dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "splits": {s: {"n": len(f),
                                "vision_coverage": round(float(f.has_vision.mean()), 4)}
                            for s, f in feats.items()}}

    print("[eval] ablations + baselines ...")
    # The train class prior lives in every head checkpoint, so the majority
    # baseline does not require the (large) train feature cache to be present.
    _, _ck = load_head("fused")
    train_counts = np.array(_ck["class_counts"], dtype=float)
    res["ablations"] = ablations.run(feats, train_counts)

    print("[eval] calibration ...")
    res["calibration"] = calibration_section(feats)

    print("[eval] streaming coverage curve ...")
    res["streaming"] = {"coverage_curve": streaming.coverage_curve(feats["test"])}

    print("[eval] commit-vs-wait policy ...")
    try:
        res["streaming"]["commit_policy"] = streaming.learn_commit_policy(
            feats["dev"], feats["test"], lam_delay=a.lambda_delay)
    except FileNotFoundError as e:
        res["streaming"]["commit_policy"] = {"skipped": str(e)}
        print(f"  [eval] commit policy skipped: {e}")

    print("[eval] belief smoothing ...")
    res["smoothing"] = smoothing.run(feats["test"])

    print("[eval] counterfactual grounding ...")
    res["counterfactual"] = counterfactual.run(feats["test"])

    if not a.no_latency:
        print("[eval] latency benchmark (real models) ...")
        from . import latency
        try:
            res["latency"] = latency.run(with_llm=not a.no_llm)
        except (SystemExit, Exception) as e:
            # A missing benchmark must not cost us every other number.
            print(f"  [eval] latency skipped: {type(e).__name__}: {e}")

    print("[eval] parameter ledger ...")
    from ..ledger import collect
    res["ledger"] = collect(include_llm=not a.no_llm)

    (ARTIFACTS / "results.json").write_text(json.dumps(res, indent=2))
    figs = plots.all_figures(res)
    print(f"\n[eval] wrote artifacts/results.json and {len(figs)} figures "
          f"in {time.time()-t0:.0f}s\n")
    print(summary(res))
    return 0


def summary(res: dict) -> str:
    ab, L = res["ablations"], []
    L.append("=" * 78)
    L.append("MELD test (2,610 utterances) — weighted F1 is the headline metric")
    L.append("=" * 78)
    L.append(f"{'config':<14}{'wF1':>8}{'macroF1':>10}{'acc':>8}{'ECE':>8}   note")
    maj = ab["majority"]
    L.append(f"{'majority':<14}{maj['weighted_f1']:>8.4f}{maj['macro_f1']:>10.4f}"
             f"{maj['accuracy']:>8.4f}{'-':>8}   always '{maj['predicts']}' "
             f"({maj['test_base_rate']*100:.1f}% of test)")
    for n in ("vision", "text_ctx", "text_solo", "text_pair",
              "fused_concat", "fused"):
        c = ab["configs"][n]
        L.append(f"{n:<14}{c['weighted_f1']:>8.4f}{c['macro_f1']:>10.4f}"
                 f"{c['accuracy']:>8.4f}{c['ece']:>8.4f}")
    bt = ab["best_text_only_config"]
    d = ab["configs"]["fused"]["weighted_f1"] - ab["configs"][bt]["weighted_f1"]
    L.append("")
    L.append(f"fusion delta over best text-only ({bt}): {d:+.4f} weighted F1")
    amb = ab["text_ambiguous_subset"]
    L.append(f"on the text-ambiguous 30% (n={amb['n']}): "
             f"text {amb['text_only']['weighted_f1']:.4f} -> "
             f"fused {amb['fused']['weighted_f1']:.4f} "
             f"({amb['fused']['weighted_f1']-amb['text_only']['weighted_f1']:+.4f})")

    cal = res["calibration"]
    L.append("")
    L.append(f"calibration: ECE {cal['ece_uncalibrated']:.4f} -> "
             f"{cal['ece_calibrated']:.4f} (T={cal['temperature']:.3f}); "
             f"at tau={cal['tau_abstain']} the lamp abstains on "
             f"{cal['abstain_rate_at_shipped_tau']*100:.1f}% and scores "
             f"{cal['at_shipped_tau']['weighted_f1']:.4f} on the rest")

    sm = res["smoothing"]
    L.append(f"inertia:     {sm['raw']['switches_per_minute']:.1f} -> "
             f"{sm['smoothed']['switches_per_minute']:.1f} switches/min "
             f"({sm['switch_reduction_pct']:.0f}% fewer), "
             f"wF1 {sm['weighted_f1_delta']:+.4f}")

    cv = res["streaming"]["coverage_curve"]
    L.append(f"streaming:   within 1 F1 point of the full utterance after "
             f"{cv['frac_to_within_1pt']*100:.0f}% heard "
             f"(~{cv['mean_utterance_ms']*cv['frac_to_within_1pt']:.0f} ms of a "
             f"{cv['mean_utterance_ms']:.0f} ms mean utterance)")

    cp = res["streaming"].get("commit_policy", {})
    if "learned_policy" in cp:
        lp, bf = cp["learned_policy"], cp["best_fixed_baseline"]
        L.append(f"commit:      learned reward {lp['mean_reward']:.4f} "
                 f"(heard {lp['mean_heard']*100:.0f}%) vs best fixed "
                 f"'{bf['name']}' {bf['mean_reward']:.4f} "
                 f"(heard {bf['mean_heard']*100:.0f}%)  "
                 f"delta {cp['reward_gain_vs_best_fixed']:+.4f}")

    cf = res["counterfactual"]
    L.append(f"grounding:   swapping the video moves the posterior by TV="
             f"{cf['vision_swap']['mean_tv']:.3f} and changes the behaviour "
             f"intent on {cf['vision_swap']['behaviour_intent_change_rate']*100:.1f}% "
             f"of utterances (text swap: TV={cf['text_swap']['mean_tv']:.3f})")

    if "latency" in res:
        st = res["latency"]["stages"]
        L.append(f"latency:     reflexive p95 "
                 f"{st['reflex_total_ms']['p95']:.0f} ms "
                 f"(budget {res['latency']['budgets']['reflexive_ms']}), "
                 + (f"TTFT p95 {st['ttft_ms']['p95']:.0f} ms "
                    f"(budget {res['latency']['budgets']['ttft_ms']})"
                    if "ttft_ms" in st else "TTFT n/a")
                 + f", RTF p95 {res['latency']['real_time_factor']['p95']:.3f}, "
                   f"peak RSS {res['latency']['peak_rss_mb']:.0f} MB")

    lg = res["ledger"]
    L.append(f"parameters:  {lg['total_params']:,} / {lg['budget']:,} "
             f"({lg['headroom_pct']}% headroom)")
    L.append("=" * 78)
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())

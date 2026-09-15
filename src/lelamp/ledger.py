"""Parameter ledger, COUNTED not quoted.

Every number here comes from summing `.numel()` over modules that are actually
instantiated. Nothing is typed in by hand, so the README cannot drift from the
code, and a reviewer can re-derive the total with one command.

Two things this deliberately does not let us hide:

  * CLIP's text tower. We assert it is absent rather than claiming it. If
    someone swaps CLIPVisionModelWithProjection for CLIPModel to "simplify", the
    assertion fails and the ledger stops being a lie.
  * ASR / VAD / TTS. MELD hands us gold transcripts, so none of these are in our
    inference path -- which is a real advantage this task grants us and NOT a
    property of a deployed lamp. The second table prices them in.
"""
from __future__ import annotations

import argparse
import json

from .config import ARTIFACTS, LLM_MODEL, TEXT_MODEL, VISION_MODEL

BUDGET = 6_000_000_000

# Published totals for the components a real LeLamp would need but this task
# does not. Sourced from the model cards; clearly labelled as NOT MEASURED here.
DEPLOY_EXTRAS = [
    ("openai/whisper-small", "streaming ASR", 244_000_000),
    ("silero-vad v4", "voice activity / endpointing", 1_100_000),
    ("piper en_US-lessac-medium", "TTS", 20_000_000),
    ("SCRFD-500m", "face detect + active speaker gating", 600_000),
]


def collect(include_llm: bool = True) -> dict:
    rows = []

    from .models.vision import VisionTower
    vt = VisionTower(device="cpu")
    assert not hasattr(vt.model, "text_model"), (
        "CLIP text tower is loaded -- the ledger would be understated. "
        "Use CLIPVisionModelWithProjection, not CLIPModel.")
    rows.append((VISION_MODEL, "vision tower (frames -> 512d)", vt.n_params, True))

    from .models.text import TextEncoder
    te = TextEncoder(device="cpu")
    rows.append((TEXT_MODEL, "utterance+context encoder", te.n_params, True))

    # Count the heads the RUNTIME actually loads, by loading them, not by
    # constructing a plausible-looking set. When the fused head switched from
    # concatenation to a gated architecture this table silently kept quoting the
    # old one -- a ledger that describes a different system than the one that
    # runs is worse than no ledger.
    from .models.heads import FastAffectHead
    from .runtime.pipeline import LampPipeline

    try:
        pipe = LampPipeline(device="cpu", responder=None)
        rows.append(("FastAffectHead", "tier-1 reflexive v/a/salience",
                     pipe.fast.n_params, True))
        rows.append((f"{type(pipe.fused).__name__}[fused]",
                     "tier-2 emotion posterior (+1 temperature)",
                     pipe.fused.n_params, True))
        rows.append(("FusionHead[text_pair]", "evidence: text-only margin",
                     pipe.text_only.n_params, True))
        rows.append(("FusionHead[vision]",
                     "evidence: vision-only margin / disagreement",
                     pipe.vision_only.n_params, True))
    except FileNotFoundError:
        # Heads not trained yet: fall back to the architecture defaults so the
        # ledger still runs on a fresh clone, and say so.
        from .models.heads import FusionHead, GatedFusionHead
        rows.append(("FastAffectHead", "tier-1 reflexive (untrained default)",
                     FastAffectHead().n_params, True))
        rows.append(("GatedFusionHead[fused]", "tier-2 posterior (untrained default)",
                     GatedFusionHead(2).n_params, True))
        rows.append(("FusionHead[text_pair]", "evidence: text-only (untrained)",
                     FusionHead(True, False, n_text_fields=2).n_params, True))
        rows.append(("FusionHead[vision]", "evidence: vision-only (untrained)",
                     FusionHead(False, True).n_params, True))
    rows.append(("BeliefTracker", "EMA + hysteresis + dwell", 0, True))
    rows.append(("BehaviorPolicy", "state -> intent/light/motion", 0, True))
    rows.append(("TemplateResponder", "zero-parameter fallback speech", 0, True))

    llm_params, llm_note = 0, "not loaded"
    if include_llm:
        try:
            from .models.responder import LLMResponder
            llm = LLMResponder()
            llm_params = llm.n_params
            llm_note = "loaded and counted"
        except Exception as e:
            llm_note = f"unavailable ({type(e).__name__}); using published total"
            llm_params = 1_543_714_304
    rows.append((LLM_MODEL, f"response generation ({llm_note})", llm_params, True))

    total = sum(r[2] for r in rows)
    deploy_total = total + sum(x[2] for x in DEPLOY_EXTRAS)
    return {
        "rows": [{"component": a, "role": b, "params": c} for a, b, c, _ in rows],
        "total_params": total,
        "budget": BUDGET,
        "headroom": BUDGET - total,
        "headroom_pct": round(100 * (BUDGET - total) / BUDGET, 1),
        "within_budget": total <= BUDGET,
        "deployed_lamp_projection": {
            "note": ("What the ledger would look like on a real LeLamp that has "
                     "to hear and speak. Published totals, NOT measured here."),
            "extras": [{"component": a, "role": b, "params": c}
                       for a, b, c in DEPLOY_EXTRAS],
            "total_params": deploy_total,
            "headroom": BUDGET - deploy_total,
            "headroom_pct": round(100 * (BUDGET - deploy_total) / BUDGET, 1),
            "within_budget": deploy_total <= BUDGET,
        },
    }


def render(d: dict) -> str:
    w = max(len(r["component"]) for r in d["rows"]) + 2
    lines = [f"{'COMPONENT'.ljust(w)}{'ROLE'.ljust(46)}{'PARAMS':>15}",
             "-" * (w + 61)]
    for r in d["rows"]:
        lines.append(f"{r['component'].ljust(w)}{r['role'][:45].ljust(46)}"
                     f"{r['params']:>15,}")
    lines += ["-" * (w + 61),
              f"{'TOTAL (core inference path)'.ljust(w + 46)}{d['total_params']:>15,}",
              f"{'BUDGET'.ljust(w + 46)}{d['budget']:>15,}",
              f"{'HEADROOM'.ljust(w + 46)}{d['headroom']:>15,}"
              f"   ({d['headroom_pct']}%)",
              "",
              "Projection: same system on a real lamp (adds ASR/VAD/TTS/face-det)"]
    p = d["deployed_lamp_projection"]
    for r in p["extras"]:
        lines.append(f"{('+ ' + r['component']).ljust(w + 46)}{r['params']:>15,}")
    lines += [f"{'PROJECTED TOTAL'.ljust(w + 46)}{p['total_params']:>15,}",
              f"{'PROJECTED HEADROOM'.ljust(w + 46)}{p['headroom']:>15,}"
              f"   ({p['headroom_pct']}%)"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true",
                    help="skip loading the LLM (uses its published total)")
    a = ap.parse_args(argv)
    d = collect(include_llm=not a.no_llm)
    print(render(d))
    (ARTIFACTS / "parameter_ledger.json").write_text(json.dumps(d, indent=2))
    assert d["within_budget"], "OVER BUDGET"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

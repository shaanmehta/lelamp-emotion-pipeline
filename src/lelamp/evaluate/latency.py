"""Measured latency, on this machine, with the real models in the loop.

Not estimated, not extrapolated from FLOPs. Every number here is a wall-clock
percentile over repeated runs of the actual code path, including the mp4 decode,
because on a robot the decode is not free either.

Budgets being checked (justified in the README):
  * reflexive tier   <= 300 ms p95 from frame arrival to emitted behaviour
  * deliberative     <= 800 ms p95 from utterance end to first spoken token
"""
from __future__ import annotations

import platform
import time

import numpy as np

from ..config import BUDGET_REFLEX_MS, BUDGET_TTFT_MS, CLIPS, DEVICE


def _pct(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    return {"n": len(a), "p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2),
            "mean": round(float(a.mean()), 2), "max": round(float(a.max()), 2)}


def _rss_mb() -> float:
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)


def run(n_iter: int = 40, with_llm: bool = True, device: str = DEVICE) -> dict:
    from ..data.video import decode_frames
    from ..models.heads import pool_frames
    from ..models.text import TextEncoder
    from ..models.vision import VisionTower
    from ..runtime.pipeline import LampPipeline

    clips = sorted(CLIPS.glob("*.mp4"))
    if not clips:
        raise SystemExit("no clips in assets/clips -- run `make features` first")
    blobs = [p.read_bytes() for p in clips[:12]]

    # Phase 1 is torch/MPS only. The responder (MLX/Metal) is loaded afterwards,
    # in phase 2, and never interleaved: two Metal clients contending inside one
    # process is the same failure mode that wedged feature extraction across two
    # processes, and it is not worth finding out the hard way twice.
    tower = VisionTower(device=device)
    text = TextEncoder(device=device)
    pipe = LampPipeline(device=device, responder=None)

    states: list[dict] = []
    stages: dict[str, list[float]] = {k: [] for k in (
        "decode_mp4_ms", "clip_1frame_ms", "clip_8frame_ms", "roberta_ms",
        "pool_ms", "fusion_heads_ms", "belief_ms", "policy_ms",
        "reflex_total_ms", "deliberative_total_ms", "ttft_ms")}
    rtf = []

    # warm-up: first call on MPS pays shader compilation, which is not a
    # steady-state cost and would poison the percentiles.
    f0, _ = decode_frames(blobs[0])
    tower.encode(f0[:1]); tower.encode(f0); text.encode(["warm up the encoder"])
    pipe.encode_text(text, "X", "warm up", [])

    for i in range(n_iter):
        blob = blobs[i % len(blobs)]

        t = time.perf_counter()
        frames, ftimes = decode_frames(blob)
        stages["decode_mp4_ms"].append((time.perf_counter() - t) * 1000)
        if len(frames) == 0:
            continue

        t = time.perf_counter()
        one = tower.encode(frames[:1])
        stages["clip_1frame_ms"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        allf = tower.encode(frames)
        dt_all = (time.perf_counter() - t) * 1000
        stages["clip_8frame_ms"].append(dt_all)

        # --- reflexive tier ------------------------------------------------
        t = time.perf_counter()
        ev = pipe.reflex(one[0], t_ms=200.0, uid="bench", dialogue_id="bench",
                         frames_seen=1, text_frac=0.3)
        stages["reflex_total_ms"].append(
            (time.perf_counter() - t) * 1000 + stages["clip_1frame_ms"][-1])

        # --- deliberative tier ---------------------------------------------
        t = time.perf_counter()
        temb = pipe.encode_text(text, "Ross",
                                "I can't believe you did that to me.",
                                ["Monica: You promised you would call."])
        stages["roberta_ms"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        pool_frames(allf)
        stages["pool_ms"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        ev2 = pipe.deliberate(temb, allf, t_ms=3000.0, uid="bench",
                              dialogue_id="bench", text_frac=1.0)
        d_ms = (time.perf_counter() - t) * 1000
        stages["deliberative_total_ms"].append(d_ms)
        stages["fusion_heads_ms"].append(ev2.latency.get("heads_ms", 0.0))
        stages["belief_ms"].append(ev2.latency.get("belief_ms", 0.0))
        stages["policy_ms"].append(ev2.latency.get("policy_ms", 0.0))

        # Real-time factor for the perception loop: compute spent per second of
        # incoming video. Must be well under 1 to consume a live stream.
        clip_s = max(len(frames) / 5.0, 0.5)
        rtf.append((stages["decode_mp4_ms"][-1] + dt_all + d_ms) / 1000.0 / clip_s)

        states.append(ev2.to_dict())

    # ---- phase 2: response generation, torch work already finished ----------
    responder_name = "template (0 params)"
    if with_llm:
        import gc
        if device == "mps":
            import torch
            torch.mps.empty_cache()
        gc.collect()
        from ..models.responder import get_responder
        responder = get_responder(prefer_llm=True)
        responder_name = (type(responder).__name__ == "LLMResponder"
                          and "Qwen2.5-1.5B-Instruct-4bit (MLX)"
                          or "template (0 params)")
        # One warm-up: the first MLX call pays lazy weight load + shader compile,
        # which is not a steady-state cost and would poison the percentiles.
        if states:
            responder([], "Ross", "warm up", states[0])
        # 15 generations is enough for a stable p95 and keeps `make eval` short;
        # each one is a full ~28-token decode.
        for st in states[:15]:
            r = responder(["Monica: You promised you would call."], "Ross",
                          "I can't believe you did that to me.", st)
            stages["ttft_ms"].append(r.first_token_ms)
            stages.setdefault("response_total_ms", []).append(r.total_ms)

    out = {
        "hardware": {
            "machine": platform.machine(), "system": platform.platform(),
            "device": device, "python": platform.python_version(),
        },
        "stages": {k: _pct(v) for k, v in stages.items() if v},
        "real_time_factor": _pct([r for r in rtf]),
        "peak_rss_mb": _rss_mb(),
        "budgets": {
            "reflexive_ms": BUDGET_REFLEX_MS,
            "ttft_ms": BUDGET_TTFT_MS,
        },
        "responder": responder_name,
    }
    out["budget_met"] = {
        "reflexive_p95": out["stages"]["reflex_total_ms"]["p95"] <= BUDGET_REFLEX_MS,
        "ttft_p95": (out["stages"]["ttft_ms"]["p95"] <= BUDGET_TTFT_MS
                     if "ttft_ms" in out["stages"] else None),
    }
    return out

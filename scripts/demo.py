"""End-to-end streaming demo: one MELD dialogue through both tiers, live.

    python scripts/demo.py                      # replay test dialogue 237
    python scripts/demo.py --speed 2            # 2x wall clock
    python scripts/demo.py --counterfactual     # same words, different video

What you're watching:
  * Frames arrive at their true timestamps. Each one is encoded by CLIP at that
    moment and drives the reflexive tier, so the lamp moves before the sentence
    is over.
  * Words arrive at a constant rate. Once coverage passes --min-coverage the
    pipeline asks "have I heard enough?" and commits as soon as calibrated
    confidence clears --commit-conf, or at the end of the utterance otherwise.
  * COMMIT_STATE is emitted before the responder runs, so the body never waits
    on the language model.

Every event gets appended to artifacts/runs/<timestamp>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lelamp.config import CLIPS, DEVICE, REFLEX_TICK_MS, RUNS  # noqa: E402
from lelamp.data.meld import load_split  # noqa: E402
from lelamp.data.video import decode_frames  # noqa: E402
from lelamp.models.responder import get_responder  # noqa: E402
from lelamp.models.text import TextEncoder  # noqa: E402
from lelamp.models.vision import VisionTower  # noqa: E402
from lelamp.runtime.pipeline import LampPipeline  # noqa: E402
from lelamp.runtime.sources import ReplayStream  # noqa: E402
from lelamp.runtime.visualizer import LampView  # noqa: E402
from lelamp.schema import round_floats  # noqa: E402


def available(dialogue: int) -> list:
    rows = [u for u in load_split("test") if u.dialogue_id == dialogue]
    rows = [u for u in rows if (CLIPS / u.clip_name).exists()]
    rows.sort(key=lambda u: u.utterance_id)
    return rows


def run_utterance(u, pipe, tower, text_enc, view, sink, speed, min_cov,
                  commit_conf, quiet=False):
    """Replay a single utterance and return the committed state event."""
    frames, ftimes = decode_frames((CLIPS / u.clip_name).read_bytes())
    words = u.text.split()
    stream = ReplayStream(duration_s=u.duration_s, n_frames=len(frames),
                          words=words, frame_times_s=list(ftimes), speed=speed)

    seen: list[np.ndarray] = []
    committed = None
    t_wall0 = time.perf_counter()

    last_reflex = -1e9
    for ev in stream:
        if committed is not None:
            continue
        if ev.kind == "frame" and len(frames):
            feat = tower.encode(frames[ev.index : ev.index + 1])[0]
            seen.append(feat)
            if ev.t_ms - last_reflex >= REFLEX_TICK_MS:
                last_reflex = ev.t_ms
                r = pipe.reflex(feat, ev.t_ms, u.uid, f"dia{u.dialogue_id}",
                                len(seen), min(1.0, (ev.t_ms / 1000) / max(u.duration_s, .2)))
                sink(r)
                if not quiet:
                    view.push(round_floats(r.to_dict()),
                              f"t={ev.t_ms:6.0f}ms  reflex   v={r.affect.valence:+.2f} "
                              f"a={r.affect.arousal:+.2f}  {r.behavior.intent}")

        elif ev.kind == "word":
            cov = (ev.index + 1) / max(len(words), 1)
            if cov >= min_cov and seen:
                emb = pipe.encode_text(text_enc, u.speaker, u.text,
                                       u.context, cov)
                p, conf = pipe.probe(emb, np.stack(seen))
                if conf >= commit_conf:
                    committed = _commit(u, pipe, emb, seen, ev.t_ms, cov, view,
                                        sink, t_wall0, quiet,
                                        why=f"confidence {conf:.2f} >= {commit_conf}")

        elif ev.kind == "end" and committed is None:
            emb = pipe.encode_text(text_enc, u.speaker, u.text, u.context, 1.0)
            committed = _commit(u, pipe, emb, seen, ev.t_ms, 1.0, view, sink,
                                t_wall0, quiet, why="end of utterance")
    return committed


def _commit(u, pipe, emb, seen, t_ms, cov, view, sink, t_wall0, quiet, why):
    st = pipe.deliberate(emb, np.stack(seen) if seen else np.zeros((0, 512), np.float32),
                         t_ms, u.uid, f"dia{u.dialogue_id}", cov,
                         dialogue_t_ms=pipe.dialogue_t_ms + t_ms,
                         visual_cue=_cue(seen))
    sink(st)                                   # body gets the state immediately
    if not quiet:
        view.push(round_floats(st.to_dict()),
                  f"t={t_ms:6.0f}ms  COMMIT   {st.affect.emotion} "
                  f"{st.affect.confidence:.2f} -> {st.behavior.intent}  ({why})")
    pipe.speak(st, u.context, u.speaker, u.text)   # ...then we talk
    sink(st)
    if not quiet:
        view.push(round_floats(st.to_dict()),
                  f"           speech   {st.latency.get('responder','-')} "
                  f"ttft={st.latency.get('ttft_ms', 0):.0f}ms")
    return st


def _cue(seen: list[np.ndarray]) -> str:
    """A coarse, honest description of what vision contributed. We do not run a
    face detector, so this describes frame dynamics, not an expression."""
    if not seen:
        return "no video"
    if len(seen) < 2:
        return "single frame, static shot"
    d = float(np.linalg.norm(seen[-1] - seen[0]))
    return ("high visual change across the utterance" if d > 0.55
            else "moderate visual change" if d > 0.3
            else "visually static shot")


def counterfactual(args, pipe, tower, text_enc):
    """Identical transcript, different video -> different state and behaviour.

    Rather than hardcoding an index and hoping, we search every (transcript,
    donor video) pair available in the vendored clips and show the one where
    swapping the video moves the posterior furthest. Cherry-picking a single
    flattering example would prove nothing, which is why the aggregate version
    over all 2,610 test utterances is the number that actually gets reported --
    this is the illustration of that number, not the evidence for it.
    """
    rows = available(args.dialogue) + available(60 if args.dialogue != 60 else 237)
    if len(rows) < 2:
        raise SystemExit("need at least two vendored clips")

    # Encode each clip's frames once.
    feats = {}
    for u in rows:
        fr, _ = decode_frames((CLIPS / u.clip_name).read_bytes())
        if len(fr):
            feats[u.uid] = tower.encode(fr)
    rows = [u for u in rows if u.uid in feats]

    best = None
    for base in rows:
        emb = pipe.encode_text(text_enc, base.speaker, base.text, base.context, 1.0)
        own = None
        for donor in rows:
            pipe.reset_dialogue()
            st = pipe.deliberate(emb, feats[donor.uid], 3000.0, base.uid,
                                 f"dia{base.dialogue_id}", 1.0,
                                 visual_cue=_cue(list(feats[donor.uid])))
            if donor.uid == base.uid:
                own = st
                continue
            if own is None:
                continue
            tv = 0.5 * sum(abs(own.affect.probs[k] - st.affect.probs[k])
                           for k in own.affect.probs)
            changed = own.behavior.intent != st.behavior.intent
            score = tv + (1.0 if changed else 0.0)
            if best is None or score > best[0]:
                best = (score, base, donor, own, st, tv, changed)

    score, base, donor, own, swapped, tv, changed = best
    pipe.reset_dialogue()
    pipe.speak(own, base.context, base.speaker, base.text)
    pipe.reset_dialogue()
    pipe.speak(swapped, base.context, base.speaker, base.text)

    print("\n" + "=" * 78)
    print("COUNTERFACTUAL GROUNDING - identical transcript, different video")
    print("=" * 78)
    print(f'transcript : {base.speaker}: "{base.text}"')
    print(f"gold label : {base.emotion}")
    print(f"(strongest pair among {len(rows)} vendored clips)\n")
    for tag, src, st in (("its own video", base, own),
                         (f"video swapped in from {donor.uid}", donor, swapped)):
        print(f"--- {tag}  (clip gold: {src.emotion}) ---")
        print(f"  emotion   : {st.affect.emotion}  conf={st.affect.confidence:.3f} "
              f"v={st.affect.valence:+.2f} a={st.affect.arousal:+.2f}")
        print(f"  intent    : {st.behavior.intent}  (priority {st.behavior.priority}, "
              f"hue {st.behavior.light.hue_deg:.0f} deg, {st.behavior.motion.posture}, "
              f"hold {st.behavior.hold_ms} ms)")
        print(f"  says      : \"{st.speech.text if st.speech else ''}\"")
        print(f"  evidence  : dominant={st.evidence.dominant_modality} "
              f"agreement={st.evidence.modality_agreement:.2f} "
              f"visual={st.evidence.visual_cue}")
    print(f"\nposterior total-variation between the two : {tv:.3f}")
    print(f"behaviour intent changed                  : {changed}")
    print("\nThis is an illustration, not evidence. The evidence is the aggregate")
    print("swap over all 2,610 test utterances: artifacts/results.json ->")
    print("counterfactual, and figure 07.")
    print("=" * 78)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dialogue", type=int, default=237)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--commit-conf", type=float, default=0.60)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--counterfactual", action="store_true")
    ap.add_argument("--cf-index", type=int, default=6)
    ap.add_argument("--device", default=DEVICE)
    a = ap.parse_args(argv)

    rows = available(a.dialogue)
    if not rows:
        raise SystemExit(
            f"No vendored clips for dialogue {a.dialogue}. "
            f"Run `make features` (streams MELD) or pick a dialogue in "
            f"assets/clips/.")

    tower = VisionTower(device=a.device)
    text_enc = TextEncoder(device=a.device)
    responder = get_responder(prefer_llm=not a.no_llm)
    pipe = LampPipeline(device=a.device, responder=responder)

    if a.counterfactual:
        counterfactual(a, pipe, tower, text_enc)
        return 0

    run_path = RUNS / f"demo_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    fh = run_path.open("w")

    def sink(ev):
        fh.write(json.dumps(round_floats(ev.to_dict())) + "\n")

    pipe.reset_dialogue()
    with LampView(f"LeLamp — MELD test dialogue {a.dialogue} "
                  f"({len(rows)} turns, {a.speed}x)") as view:
        for u in rows:
            view.log.append(f"\n[{u.speaker}] \"{u.text[:70]}\"  (gold: {u.emotion})")
            run_utterance(u, pipe, tower, text_enc, view, sink, a.speed,
                          a.min_coverage, a.commit_conf)
            pipe.advance_dialogue(u.duration_s * 1000)
    fh.close()
    evs = [json.loads(l) for l in run_path.read_text().splitlines()]
    n_reflex = sum(e["tier"] == "reflexive" for e in evs)
    n_commit = sum(e["tier"] == "deliberative" and e["speech"] is None for e in evs)
    print(f"\n{len(evs)} events -> {run_path}")
    print(f"  {n_reflex} reflexive (pre-endpoint), {n_commit} commits, "
          f"{len(evs)-n_reflex-n_commit} with speech attached")
    with_speech = [e for e in evs if e["tier"] == "deliberative" and e["speech"]]
    if with_speech:
        e = with_speech[0]
        print("\nfirst COMMIT_STATE, after speech was attached "
              "(the body already received the identical event, minus `speech`, "
              f"{e['latency'].get('ttft_ms', 0):.0f} ms earlier):")
        print(json.dumps(e, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

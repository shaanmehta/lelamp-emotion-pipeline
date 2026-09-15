"""The two-tier orchestrator.

Tier 1, reflexive, budget 300ms p95. Every REFLEX_TICK_MS the newest CLIP frame
goes through FastAffectHead to valence/arousal/salience and then to behaviour.
This runs while the person is still talking, which is the whole reason vision is
worth its compute on a robot: it's the only signal available before the
utterance ends.

Tier 2, deliberative, budget 800ms p95 to first token. At commit, pooled frames
and the text embeddings go through the fusion head, then calibration, then
belief smoothing, then the behaviour policy. COMMIT_STATE is emitted before the
responder runs, so the body never waits on the language model.

Three heads get loaded, not one: fused, text-only, vision-only. The two
single-modality heads aren't decoration. Their disagreement is what separates
"I'm unsure because the evidence is weak" (attend) from "I'm unsure because my
two senses contradict each other" (confused_tilt). They cost about 1.9M
parameters between them and they're in the ledger.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..config import (BUDGET_REFLEX_MS, CONTEXT_TURNS, DEVICE, EMOTIONS, HEADS,
                      REFLEX_TICK_MS, TAU_ABSTAIN)
from ..models.heads import (FastAffectHead, FusionHead,
                            GatedFusionHead, pool_frames)
from ..schema import (Affect, Behavior, Belief, Coverage, Evidence, Speech,
                      StateEvent)
from .belief import BeliefTracker
from .policy import decide, project_va


def _load_fusion(name: str, device: str) -> tuple[FusionHead, dict]:
    ck = torch.load(HEADS / f"{name}.pt", map_location=device, weights_only=False)
    if ck.get("gated"):
        m = GatedFusionHead(n_text_fields=ck["n_text_fields"])
    else:
        m = FusionHead(use_text=ck["use_text"], use_vision=ck["use_vision"],
                       n_text_fields=ck.get("n_text_fields", 1))
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck


@dataclass
class Stage:
    """Per-stage latency accounting. Measured, never estimated."""
    marks: dict = field(default_factory=dict)
    _t: float = 0.0

    def start(self):
        self._t = time.perf_counter()
        return self

    def lap(self, name: str):
        now = time.perf_counter()
        self.marks[name] = (now - self._t) * 1000.0
        self._t = now
        return self.marks[name]

    @property
    def total_ms(self) -> float:
        return float(sum(self.marks.values()))


class LampPipeline:
    def __init__(self, device: str = DEVICE, responder=None,
                 tau: float = TAU_ABSTAIN, smooth: bool = True):
        self.device = device
        self.fused, ck = _load_fusion("fused", device)
        # Must consume the SAME text fields as the fused head, or it cannot be
        # fed the same tensor. `text_pair` is the matching single-modality head.
        self.text_only, _ = _load_fusion("text_pair", device)
        self.vision_only, _ = _load_fusion("vision", device)

        fck = torch.load(HEADS / "fast_affect.pt", map_location=device,
                         weights_only=False)
        self.fast = FastAffectHead()
        self.fast.load_state_dict(fck["state_dict"])
        self.fast = self.fast.to(device).eval()

        # Which text encodings the fused head expects. Callers must not guess:
        # every place that guessed ("just use text_ctx") broke silently the
        # moment the head started consuming two encodings.
        self.text_fields: list[str] = list(ck["text_fields"])
        self.va_offset = np.array(ck["va_offset"], dtype=np.float64)
        self.tau, self.smooth = tau, smooth
        self.belief = BeliefTracker()
        self.responder = responder
        self.reset_dialogue()

    # ------------------------------------------------------------------ util
    def reset_dialogue(self):
        self.belief.reset()
        self._t_dialogue_ms = 0.0

    @property
    def dialogue_t_ms(self) -> float:
        """Elapsed speech time in the current dialogue. Belief hysteresis and
        minimum-dwell are defined against this clock, not the per-utterance one,
        or every utterance would reset its own dwell timer."""
        return self._t_dialogue_ms

    def advance_dialogue(self, ms: float):
        self._t_dialogue_ms += ms

    @property
    def n_params(self) -> int:
        return (self.fused.n_params + self.text_only.n_params
                + self.vision_only.n_params + self.fast.n_params)

    def encode_text(self, encoder, speaker: str, utterance: str,
                    context: list[str], frac: float = 1.0) -> np.ndarray:
        """Build the fused head's text input with the right encodings, in order.

        `text_solo` is the utterance alone; `text_ctx` prepends up to three
        prior turns. Both are truncated to `frac` of the current utterance so
        this works mid-stream.
        """
        from ..models.text import prefix_text

        head = " </s> ".join(context[-CONTEXT_TURNS:]) if context else ""
        cur = f"{speaker}: {prefix_text(utterance, frac)}"
        variants = {
            "text_solo": cur,
            "text_ctx": f"{head} </s> {cur}" if head else cur,
        }
        wanted = [variants[f] for f in self.text_fields]
        embs = encoder.encode(wanted)          # one batched forward, not N
        return np.concatenate(embs, axis=-1)

    # --------------------------------------------------------------- tier 1
    @torch.inference_mode()
    def reflex(self, frame_feat: np.ndarray, t_ms: float, uid: str,
               dialogue_id: str, frames_seen: int, text_frac: float) -> StateEvent:
        st = Stage().start()
        x = torch.from_numpy(np.asarray(frame_feat, dtype=np.float32)[None]).to(self.device)
        v, a, sal = self.fast(x)[0].tolist()
        st.lap("fast_head_ms")

        # Salience doubles as the reflexive tier's confidence: it is literally
        # "is anything emotional happening", which is the only question one
        # frame can answer.
        beh = decide(np.full(len(EMOTIONS), 1 / len(EMOTIONS)),
                     confidence=float(sal), valence=float(v), arousal=float(a),
                     tau=0.5)
        st.lap("policy_ms")

        ev = StateEvent(
            utterance_id=uid, dialogue_id=dialogue_id, tier="reflexive",
            t_emit_ms=round(t_ms, 1),
            coverage=Coverage(text_frac=round(text_frac, 3), frames_seen=frames_seen),
            affect=Affect(emotion=None, probs=None, confidence=round(float(sal), 4),
                          calibrated=False, valence=round(float(v), 4),
                          arousal=round(float(a), 4), uncertain=float(sal) < 0.5),
            behavior=beh, latency=dict(st.marks, budget_ms=BUDGET_REFLEX_MS),
        )
        return ev

    # --------------------------------------------------------------- tier 2
    @torch.inference_mode()
    def deliberate(self, text_emb: np.ndarray, frame_feats: np.ndarray,
                   t_ms: float, uid: str, dialogue_id: str, text_frac: float,
                   dialogue_t_ms: float | None = None,
                   visual_cue: str = "n/a") -> StateEvent:
        st = Stage().start()
        vis = torch.from_numpy(pool_frames(frame_feats)[None]).to(self.device)
        txt = torch.from_numpy(np.asarray(text_emb, dtype=np.float32)[None]).to(self.device)
        st.lap("pool_ms")

        p = self.fused.probs(txt, vis)[0].cpu().numpy()
        pt = self.text_only.probs(txt, None)[0].cpu().numpy()
        pv = self.vision_only.probs(None, vis)[0].cpu().numpy()
        st.lap("heads_ms")

        t_dia = dialogue_t_ms if dialogue_t_ms is not None else t_ms
        if self.smooth:
            b = self.belief.update(p, t_dia)
            probs, emotion = b.probs, b.emotion
        else:
            probs, emotion = p, EMOTIONS[int(p.argmax())]
            b = None
        st.lap("belief_ms")

        conf = float(probs[EMOTIONS.index(emotion)])
        v, a = project_va(probs, self.va_offset)
        agreement = float(1.0 - 0.5 * np.abs(pt - pv).sum())   # 1 - total variation
        beh = decide(probs, conf, v, a, agreement=agreement, tau=self.tau)
        st.lap("policy_ms")

        def margin(q):
            s = np.sort(q)[::-1]
            return float(s[0] - s[1])

        uncertain = conf < self.tau
        return StateEvent(
            utterance_id=uid, dialogue_id=dialogue_id, tier="deliberative",
            t_emit_ms=round(t_ms, 1),
            coverage=Coverage(text_frac=round(text_frac, 3),
                              frames_seen=int(len(frame_feats))),
            affect=Affect(
                emotion=emotion,
                probs={e: round(float(x), 4) for e, x in zip(EMOTIONS, probs)},
                confidence=round(conf, 4), calibrated=True,
                valence=round(v, 4), arousal=round(a, 4), uncertain=uncertain),
            behavior=beh,
            evidence=Evidence(
                dominant_modality=("text" if margin(pt) > margin(pv) else "vision"),
                text_margin=round(margin(pt), 4),
                vision_margin=round(margin(pv), 4),
                modality_agreement=round(agreement, 4),
                visual_cue=visual_cue),
            belief=None if b is None else Belief(
                smoothing=self.belief.description, raw_emotion=b.raw_emotion,
                prev_emotion=b.prev_emotion, switched=b.switched,
                dwell_ms=b.dwell_ms, suppressed_switches=b.suppressed),
            latency=dict(st.marks),
        )

    @torch.inference_mode()
    def probe(self, text_emb: np.ndarray, frame_feats: np.ndarray) -> tuple[np.ndarray, float]:
        """Calibrated posterior WITHOUT touching belief state.

        Used by the commit-vs-wait logic to ask "have I heard enough yet?"
        repeatedly during an utterance. Folding these probes into the belief EMA
        would let a single utterance vote several times and quietly defeat the
        smoothing.
        """
        vis = torch.from_numpy(pool_frames(frame_feats)[None]).to(self.device)
        txt = torch.from_numpy(np.asarray(text_emb, dtype=np.float32)[None]).to(self.device)
        p = self.fused.probs(txt, vis)[0].cpu().numpy()
        return p, float(p.max())

    # ------------------------------------------------------------- speech
    def speak(self, event: StateEvent, context: list[str], speaker: str,
              utterance: str) -> StateEvent:
        """Attach speech. Called AFTER the state event has already been emitted."""
        if self.responder is None:
            return event
        if event.affect.uncertain and event.behavior.intent == "idle":
            event.speech = Speech("", "silent", 0, allowed=False)
            return event
        r = self.responder(context, speaker, utterance, event.to_dict())
        event.speech = Speech(
            text=r.text,
            style={"celebrate": "bright", "soothe": "gentle", "recoil": "measured",
                   "startle": "clipped", "attend": "curious",
                   "confused_tilt": "hesitant"}.get(event.behavior.intent, "plain"),
            onset_delay_ms=int(min(400, event.behavior.hold_ms * 0.2)),
            allowed=not (event.affect.uncertain and event.behavior.intent == "idle"),
        )
        event.latency["ttft_ms"] = round(r.first_token_ms, 1)
        event.latency["response_total_ms"] = round(r.total_ms, 1)
        event.latency["responder"] = r.source
        return event

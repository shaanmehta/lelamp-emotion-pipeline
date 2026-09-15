"""Short spoken response, grounded in the structured state.

Two implementations behind one interface:

* TemplateResponder -- ZERO learned parameters. Always available, sub-millisecond.
  It exists because a lamp that misses its speaking window is worse than a lamp
  that says something simple on time, and because it gives the latency budget a
  real fallback rather than a hypothetical one.

* LLMResponder -- Qwen2.5-1.5B-Instruct, 4-bit, via MLX. Streams tokens.

The LLM never sees pixels. It is grounded in the multimodal input THROUGH the
structured state -- the emotion posterior, valence/arousal, the dominant modality
and the visual cue all go into the prompt. Whether that channel is actually live
(rather than decorative) is not something we assert; it is measured, by swapping
the visual stream under a fixed transcript across the whole test set. See
evaluate/counterfactual.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

SYSTEM = (
    "You are LeLamp, a small desk lamp robot with no face and no hands. You "
    "express yourself through light, posture and timing, and through very short "
    "spoken lines. Reply with ONE sentence of at most 14 words, in character, "
    "warm but not saccharine. You are given a perception report inferred from "
    "the speaker's words AND their video. Let it change what you say: never "
    "respond cheerfully to someone who is angry or upset. If confidence is low, "
    "say something that acknowledges you are not sure rather than guessing. "
    "Never mention the perception report, confidence numbers, or that you are a "
    "model. Never repeat, quote or paraphrase back the words they said -- "
    "RESPOND to them. Output only your one sentence."
)


@dataclass
class ResponseChunk:
    text: str
    first_token_ms: float
    total_ms: float
    source: str        # "llm" | "template"


def build_prompt(context: list[str], speaker: str, utterance: str,
                 state: dict) -> str:
    aff, beh, ev = state["affect"], state["behavior"], state.get("evidence") or {}
    convo = "\n".join(context[-3:]) if context else "(start of conversation)"
    unsure = "yes" if aff["uncertain"] else "no"
    return (
        f"Conversation so far:\n{convo}\n\n"
        f"They just said:\n{speaker}: \"{utterance}\"\n\n"
        f"Perception report (from words + video):\n"
        f"- emotion: {aff['emotion']} (confidence {aff['confidence']:.2f}, "
        f"unsure: {unsure})\n"
        f"- valence {aff['valence']:+.2f}, arousal {aff['arousal']:+.2f}\n"
        f"- visual: {ev.get('visual_cue', 'n/a')}\n"
        f"- strongest evidence came from: {ev.get('dominant_modality', 'n/a')}\n"
        f"- your body is already doing: {beh['intent']} "
        f"({beh['motion']['posture']})\n\n"
        f"Reply as LeLamp:"
    )


class TemplateResponder:
    """Deterministic, zero-parameter. Grounded via the state, not the transcript."""

    n_params = 0

    LINES = {
        "celebrate": ["Oh, that's good news — I'm glowing about it.",
                      "Yes! That deserves a little brightness."],
        "acknowledge": ["Got it. I'm with you.", "Mm. I hear that."],
        "soothe": ["That sounds heavy. I'll sit here with you.",
                   "Take your time. I'm not going anywhere."],
        "recoil": ["Okay — that landed hard. I'm listening.",
                   "Whoa. I'll give you room."],
        "startle": ["Oh! That caught me off guard.",
                    "Whoa — give me a second with that."],
        "attend": ["I'm listening — tell me more?",
                   "I'm not quite sure I follow. Go on?"],
        "confused_tilt": ["I'm getting mixed signals. Say more?",
                          "Hm — I can't quite read you there."],
        "idle": ["Mm-hm.", "Okay."],
    }

    def stream(self, context, speaker, utterance, state) -> Iterator[str]:
        yield self(context, speaker, utterance, state).text

    def __call__(self, context, speaker, utterance, state) -> ResponseChunk:
        t0 = time.perf_counter()
        intent = state["behavior"]["intent"]
        opts = self.LINES.get(intent, self.LINES["attend"])
        # Deterministic pick, but varies with the utterance so a dialogue does
        # not repeat one line verbatim.
        text = opts[hash(utterance) % len(opts)]
        ms = (time.perf_counter() - t0) * 1000
        return ResponseChunk(text, ms, ms, "template")


class LLMResponder:
    """Qwen2.5-1.5B-Instruct (4-bit, MLX). Raises at construction if unavailable."""

    def __init__(self, model_id: str | None = None, max_tokens: int = 28):
        from mlx_lm import load  # raises ImportError if mlx-lm is absent

        from ..config import LLM_MODEL

        self.model_id = model_id or LLM_MODEL
        self.model, self.tokenizer = load(self.model_id)
        self.max_tokens = max_tokens

    @property
    def n_params(self) -> int:
        """LOGICAL parameter count, with quantised weights unpacked.

        This is a trap worth being explicit about. MLX stores a 4-bit weight
        matrix packed into a uint32 array, so naively summing `array.size` over
        the loaded parameters returns ~241M for Qwen2.5-1.5B -- roughly an
        eighth of the truth, because eight 4-bit weights share each uint32.
        Reporting that number would understate the ledger by 1.3 BILLION
        parameters, in our favour.

        The constraint is on TOTAL LEARNED PARAMETERS. Quantisation changes how
        many bits a weight costs, not how many weights exist. So we unpack:
        every uint32-backed array counts 32/bits entries per element. Scales and
        biases are counted at face value -- they are learned too.
        """
        import mlx.core as mx
        from mlx.utils import tree_flatten

        bits = self._quant_bits()
        total = 0
        for _, v in tree_flatten(self.model.parameters()):
            if not hasattr(v, "size"):
                continue
            if v.dtype == mx.uint32:
                total += v.size * (32 // bits)
            else:
                total += v.size
        return total

    def _quant_bits(self) -> int:
        """Bit width from the model config; 4 if the field is absent."""
        import json

        from huggingface_hub import try_to_load_from_cache
        try:
            path = try_to_load_from_cache(self.model_id, "config.json")
            if isinstance(path, str):
                q = json.loads(open(path).read()).get("quantization") or {}
                return int(q.get("bits", 4))
        except Exception:
            pass
        return 4

    def _messages(self, context, speaker, utterance, state):
        return [{"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": build_prompt(context, speaker, utterance, state)}]

    def stream(self, context, speaker, utterance, state) -> Iterator[str]:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tokenizer.apply_chat_template(
            self._messages(context, speaker, utterance, state),
            add_generation_prompt=True, tokenize=False)
        sampler = make_sampler(temp=0.7, top_p=0.9)
        for r in stream_generate(self.model, self.tokenizer, prompt,
                                 max_tokens=self.max_tokens, sampler=sampler):
            yield r.text

    @staticmethod
    def _is_echo(reply: str, utterance: str) -> bool:
        """Small instruct models sometimes just parrot the input back.

        That is the exact anti-pattern this system is supposed to avoid -- a
        reply that would be identical without any perception at all -- so we
        detect it by token overlap and fall back rather than shipping it.
        """
        a = {w.lower().strip(".,!?\"'") for w in reply.split() if len(w) > 3}
        b = {w.lower().strip(".,!?\"'") for w in utterance.split() if len(w) > 3}
        if not a or not b:
            return False
        return len(a & b) / len(a) > 0.6

    def __call__(self, context, speaker, utterance, state) -> ResponseChunk:
        t0 = time.perf_counter()
        first, parts = None, []
        for piece in self.stream(context, speaker, utterance, state):
            if first is None:
                first = (time.perf_counter() - t0) * 1000
            parts.append(piece)
        total = (time.perf_counter() - t0) * 1000
        text = "".join(parts).strip().strip('"')
        if self._is_echo(text, utterance):
            return ResponseChunk(
                TemplateResponder()(context, speaker, utterance, state).text,
                first or total, total, "template(echo-guard)")
        return ResponseChunk(text, first or total, total, "llm")


def get_responder(prefer_llm: bool = True, verbose: bool = True):
    """LLM if we can load it, template otherwise. Never fails."""
    if prefer_llm:
        try:
            r = LLMResponder()
            if verbose:
                print(f"[responder] {r.model_id}: {r.n_params:,} params")
            return r
        except Exception as e:
            if verbose:
                print(f"[responder] LLM unavailable ({type(e).__name__}: {e}); "
                      f"using zero-parameter template responder")
    return TemplateResponder()

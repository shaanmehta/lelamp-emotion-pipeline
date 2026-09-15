"""The contract with the behaviour layer.

Someone writing LeLamp's motion code should only need four fields:
`behavior.intent`, `behavior.priority`, `behavior.hold_ms`, `behavior.decay_ms`,
plus `expressivity` as a global gain. Everything else (posteriors, evidence,
belief internals) is diagnostic. I split it that way so perception can change
how it computes emotion without breaking motion.

Two event types, one envelope, so a consumer only needs one parser:
  * tier="reflexive"    emitted every REFLEX_TICK_MS from vision alone. Carries
                        valence and arousal but no emotion category.
  * tier="deliberative" emitted once at commit. Full posterior, and speech if
                        the lamp decided to say something.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal

SCHEMA_VERSION = "1.0"

Intent = Literal[
    "idle",           # nothing worth reacting to
    "attend",         # orient toward speaker, low commitment -- the "uncertain" pose
    "acknowledge",    # small nod/pulse: I heard you
    "celebrate",      # high valence, high arousal
    "soothe",         # low valence, low arousal: slow, warm, dim
    "recoil",         # low valence, high arousal directed at us
    "startle",        # sharp onset, high arousal, valence unclear
    "confused_tilt",  # modalities disagree, or posterior is flat
]


@dataclass
class Light:
    hue_deg: float
    sat: float
    value: float
    pulse_hz: float = 0.0


@dataclass
class Motion:
    posture: str          # lean_in | lean_back | upright | droop | recoil_pose
    gaze: str             # speaker | away | down | scan
    amplitude: float      # 0..1
    speed: float          # 0..1


@dataclass
class Behavior:
    intent: str
    priority: int         # 0 idle .. 3 interrupt whatever is playing
    hold_ms: int
    decay_ms: int
    expressivity: float   # global gain, scales with confidence
    light: Light
    motion: Motion


@dataclass
class Affect:
    emotion: str | None
    probs: dict[str, float] | None
    confidence: float
    calibrated: bool
    valence: float
    arousal: float
    uncertain: bool


@dataclass
class Evidence:
    dominant_modality: str          # text | vision | tie
    text_margin: float              # p1 - p2 from the text-only head
    vision_margin: float
    modality_agreement: float       # 1 - total variation between the two heads
    visual_cue: str


@dataclass
class Speech:
    text: str
    style: str
    onset_delay_ms: int
    allowed: bool


@dataclass
class Belief:
    smoothing: str
    raw_emotion: str | None
    prev_emotion: str | None
    switched: bool
    dwell_ms: int
    suppressed_switches: int


@dataclass
class Coverage:
    text_frac: float
    frames_seen: int
    audio: bool = False


@dataclass
class StateEvent:
    utterance_id: str
    dialogue_id: str
    tier: str                     # reflexive | deliberative
    t_emit_ms: float
    coverage: Coverage
    affect: Affect
    behavior: Behavior
    evidence: Evidence | None = None
    speech: Speech | None = None
    belief: Belief | None = None
    schema_version: str = SCHEMA_VERSION
    latency: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Emit schema_version first so a reader can dispatch on it immediately.
        return {"schema_version": d.pop("schema_version"), **d}

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_round)


def _round(o):
    raise TypeError(f"not serialisable: {type(o)}")


def round_floats(o, nd: int = 4):
    """Keep emitted JSON readable; the wire format is small on purpose."""
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: round_floats(v, nd) for k, v in o.items()}
    if isinstance(o, list):
        return [round_floats(v, nd) for v in o]
    return o

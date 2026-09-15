"""Contract and runtime invariants."""
from __future__ import annotations

import json

import numpy as np

from lelamp.config import EMOTIONS
from lelamp.data.meld import load_split
from lelamp.models.text import prefix_text
from lelamp.runtime.belief import BeliefTracker
from lelamp.runtime.sources import ReplayStream


def _peaked(e, pm=0.75):
    p = np.full(len(EMOTIONS), (1 - pm) / (len(EMOTIONS) - 1))
    p[EMOTIONS.index(e)] = pm
    return p


# ---------------------------------------------------------------- belief ---
def test_single_frame_blip_is_absorbed():
    """One noisy turn must not move the lamp."""
    t = BeliefTracker()
    for e in ("anger", "anger"):
        t.update(_peaked(e), 0)
    out = t.update(_peaked("joy"), 3000)
    assert out.emotion == "anger" and out.raw_emotion == "joy"
    assert out.suppressed == 1


def test_sustained_change_does_commit():
    """Inertia must not be paralysis."""
    t = BeliefTracker()
    for i in range(3):
        t.update(_peaked("anger"), i * 1500)
    last = None
    for i in range(3, 7):
        last = t.update(_peaked("neutral"), i * 1500)
    assert last.emotion == "neutral"


def test_min_dwell_blocks_immediate_flip():
    t = BeliefTracker()
    t.update(_peaked("joy"), 0)
    out = t.update(_peaked("anger"), 100)      # 100 ms later: inside min dwell
    assert out.emotion == "joy"


def test_reset_clears_state_between_dialogues():
    t = BeliefTracker()
    t.update(_peaked("anger"), 0)
    t.reset()
    out = t.update(_peaked("joy"), 0)
    assert out.emotion == "joy" and out.prev_emotion is None


# --------------------------------------------------------------- streaming --
def test_replay_stream_is_causal_and_ordered():
    s = ReplayStream(duration_s=3.0, n_frames=8, words="a b c d e".split(),
                     realtime=False)
    evs = s.schedule()
    assert [e.t_ms for e in evs] == sorted(e.t_ms for e in evs)
    assert evs[-1].kind == "end"
    assert sum(e.kind == "frame" for e in evs) == 8
    assert sum(e.kind == "word" for e in evs) == 5
    # Nothing may be scheduled after the utterance ends.
    assert max(e.t_ms for e in evs) <= 3000.0 + 1e-6


def test_prefix_text_is_monotone():
    s = "I cannot believe you did that to me"
    prev = ""
    for f in (0.1, 0.3, 0.5, 0.8, 1.0):
        cur = prefix_text(s, f)
        assert cur.startswith(prev)
        prev = cur
    assert prefix_text(s, 1.0) == s


# ------------------------------------------------------------------ data ---
def test_context_is_strictly_causal():
    """Peeking at later turns is the easiest leak to commit by accident."""
    rows = load_split("dev")
    by_uid = {u.uid: u for u in rows}
    for u in rows[:400]:
        for c in u.context:
            speaker = c.split(":", 1)[0]
            # every context line must come from an EARLIER turn of this dialogue
            assert any(
                p.speaker == speaker and p.utterance_id < u.utterance_id
                for p in rows
                if p.dialogue_id == u.dialogue_id
            ), (u.uid, c)
    assert by_uid


def test_meld_split_sizes_match_published():
    assert len(load_split("train")) == 9989
    assert len(load_split("dev")) == 1109
    assert len(load_split("test")) == 2610


# ---------------------------------------------------------------- schema ---
def test_state_event_serialises_and_leads_with_version():
    from lelamp.runtime.policy import decide, project_va
    from lelamp.schema import Affect, Coverage, StateEvent, round_floats

    p = _peaked("joy")
    v, a = project_va(p)
    ev = StateEvent("dia1_utt0", "dia1", "deliberative", 120.0, Coverage(1.0, 8),
                    Affect("joy", {e: float(x) for e, x in zip(EMOTIONS, p)},
                           0.75, True, v, a, False),
                    decide(p, 0.75, v, a))
    d = round_floats(ev.to_dict())
    assert next(iter(d)) == "schema_version"
    json.loads(json.dumps(d))          # must be JSON-serialisable
    for k in ("intent", "priority", "hold_ms", "decay_ms", "expressivity"):
        assert k in d["behavior"], f"behaviour contract missing {k}"


def test_reflexive_event_carries_no_emotion_category():
    """Tier 1 must never commit the lamp to a 7-way label."""
    from lelamp.schema import Affect, Coverage, StateEvent
    from lelamp.runtime.policy import decide
    ev = StateEvent("u", "d", "reflexive", 200.0, Coverage(0.3, 2),
                    Affect(None, None, 0.4, False, -0.2, 0.5, True),
                    decide(np.full(7, 1 / 7), 0.4, -0.2, 0.5))
    d = ev.to_dict()
    assert d["affect"]["emotion"] is None and d["affect"]["probs"] is None

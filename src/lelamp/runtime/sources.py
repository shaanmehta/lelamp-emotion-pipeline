"""Turning a pre-segmented MELD clip back into something that arrives over time.

MELD clips are pre-segmented; real speech is not. We cannot invent the part of
the problem MELD removed, but we can stop pretending the whole utterance is
available at t=0. A ReplayStream emits frames at their true decode timestamps
and words at a constant rate derived from the clip's own duration, so the
pipeline sees partial evidence and has to decide when to commit.

Honest limits of this simulation, stated here and in the README:
  * Word timings are interpolated, not forced-aligned. MELD ships no word-level
    alignment, and the Text+Vision track has no audio to align against.
  * There is no VAD and therefore no real endpointing. The end of the clip is
    given to us. What we DO model is the commit-vs-wait decision, which is the
    half of the problem that determines how the robot behaves.

The interface is deliberately thin (three event types) so a camera/microphone
source could be dropped in without touching the pipeline. We did not ship one:
the target machine has no webcam available for this work, and an untested
capture path in a repo a reviewer runs once is a liability, not a feature.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Literal

EventKind = Literal["frame", "word", "end"]


@dataclass
class StreamEvent:
    kind: EventKind
    t_ms: float          # milliseconds since utterance onset
    index: int           # frame index or word index
    payload: object = None


@dataclass
class ReplayStream:
    """Replays one utterance at wall-clock rate (or `speed`x that)."""

    duration_s: float
    n_frames: int
    words: list[str]
    frame_times_s: list[float] | None = None
    speed: float = 1.0
    realtime: bool = True

    def schedule(self) -> list[StreamEvent]:
        dur_ms = max(self.duration_s, 0.2) * 1000.0
        events: list[StreamEvent] = []

        if self.frame_times_s is not None and len(self.frame_times_s) == self.n_frames:
            ft = [float(t) * 1000.0 for t in self.frame_times_s]
        else:
            ft = [dur_ms * (i + 0.5) / max(self.n_frames, 1)
                  for i in range(self.n_frames)]
        for i, t in enumerate(ft):
            events.append(StreamEvent("frame", min(t, dur_ms), i))

        n = len(self.words)
        for i, w in enumerate(self.words):
            # A word is "heard" when it finishes, not when it starts.
            events.append(StreamEvent("word", dur_ms * (i + 1) / max(n, 1), i, w))

        events.append(StreamEvent("end", dur_ms, -1))
        events.sort(key=lambda e: (e.t_ms, 0 if e.kind == "frame" else 1))
        return events

    def __iter__(self) -> Iterator[StreamEvent]:
        t0 = time.perf_counter()
        for ev in self.schedule():
            if self.realtime:
                target = t0 + (ev.t_ms / 1000.0) / max(self.speed, 1e-6)
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            yield ev

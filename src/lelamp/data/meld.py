"""MELD CSV loading, dialogue context assembly and split bookkeeping.

The only place that knows about MELD's on-disk naming conventions.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from pathlib import Path

from ..config import CSV_DIR, CONTEXT_TURNS, E2I

# MELD ships each split's clips in a differently-named directory inside the
# per-split tarball. Mapping kept here so nothing else has to care.
SPLIT_DIRS = {
    "train": "train_splits",
    "dev": "dev_splits_complete",
    "test": "output_repeated_splits_test",
}
SPLIT_TARS = {"train": "train.tar.gz", "dev": "dev.tar.gz", "test": "test.tar.gz"}

_SRT = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


def _srt_seconds(s: str) -> float:
    m = _SRT.search(s or "")
    if not m:
        return 0.0
    h, mi, sec, ms = (int(g) for g in m.groups())
    return h * 3600 + mi * 60 + sec + ms / 1000.0


@dataclass
class Utterance:
    uid: str            # "dia125_utt3" -- also the clip basename
    split: str
    dialogue_id: int
    utterance_id: int
    speaker: str
    text: str
    emotion: str
    label: int
    duration_s: float
    context: list[str] = field(default_factory=list)  # prior turns, oldest first

    @property
    def clip_name(self) -> str:
        return f"{self.uid}.mp4"


CSV_URL = ("https://raw.githubusercontent.com/declare-lab/MELD/master/"
           "data/MELD/{split}_sent_emo.csv")


def ensure_csv(split: str) -> "Path":
    """MELD transcripts/labels, fetched once if absent (~1.5 MB for all three).

    They are also committed to this repo so `make demo` works with no network
    at all; this is the fallback for a checkout that dropped them.
    """
    path = CSV_DIR / f"{split}_sent_emo.csv"
    if path.exists() and path.stat().st_size > 0:
        return path
    import ssl
    import urllib.request

    import certifi
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context(cafile=certifi.where())
    url = CSV_URL.format(split=split)
    print(f"[meld] fetching {split}_sent_emo.csv ...", flush=True)
    with urllib.request.urlopen(url, timeout=60, context=ctx) as r:
        path.write_bytes(r.read())
    return path


def load_split(split: str) -> list[Utterance]:
    """Read one MELD CSV into Utterance records with prior-turn context.

    Context is strictly causal: only turns with a lower Utterance_ID in the same
    dialogue. Peeking at later turns would be a leak and is the easiest one to
    commit by accident.
    """
    path = ensure_csv(split)
    rows: list[Utterance] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            emo = r["Emotion"].strip().lower()
            if emo not in E2I:
                continue
            dia, utt = int(r["Dialogue_ID"]), int(r["Utterance_ID"])
            dur = _srt_seconds(r["EndTime"]) - _srt_seconds(r["StartTime"])
            rows.append(
                Utterance(
                    uid=f"dia{dia}_utt{utt}",
                    split=split,
                    dialogue_id=dia,
                    utterance_id=utt,
                    speaker=r["Speaker"].strip(),
                    text=r["Utterance"].strip(),
                    emotion=emo,
                    label=E2I[emo],
                    # A handful of MELD rows have corrupt timestamps (end < start).
                    # Fall back to a plausible duration rather than dropping data.
                    duration_s=dur if 0.2 < dur < 30 else 2.5,
                )
            )

    by_dia: dict[int, list[Utterance]] = {}
    for u in rows:
        by_dia.setdefault(u.dialogue_id, []).append(u)
    for turns in by_dia.values():
        turns.sort(key=lambda u: u.utterance_id)
        for i, u in enumerate(turns):
            prev = turns[max(0, i - CONTEXT_TURNS) : i]
            u.context = [f"{p.speaker}: {p.text}" for p in prev]
    return rows


def load_all() -> dict[str, list[Utterance]]:
    return {s: load_split(s) for s in ("train", "dev", "test")}


def dialogues(rows: list[Utterance]) -> dict[int, list[Utterance]]:
    out: dict[int, list[Utterance]] = {}
    for u in rows:
        out.setdefault(u.dialogue_id, []).append(u)
    for v in out.values():
        v.sort(key=lambda u: u.utterance_id)
    return out


def context_block(u: Utterance) -> str:
    """Flat string handed to the text encoder: prior turns then the current one."""
    ctx = " </s> ".join(u.context)
    cur = f"{u.speaker}: {u.text}"
    return f"{ctx} </s> {cur}" if ctx else cur

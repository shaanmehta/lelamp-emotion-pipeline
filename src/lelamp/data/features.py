"""Joins the cached text and vision features into aligned arrays.

Text features exist for every MELD utterance. Vision features only exist where
the clip actually decoded. I keep an explicit `has_vision` mask instead of
quietly dropping rows, for two reasons: how many clips failed to decode is a
number the README should report, and every ablation has to run on the same rows
or the comparison means nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import FEATURES, N_EMO
from ..models.heads import VIS_POOL_DIM, pool_frames


@dataclass
class SplitFeatures:
    uids: np.ndarray
    text_ctx: np.ndarray      # (N,768) with up to 3 prior turns
    text_solo: np.ndarray     # (N,768) utterance in isolation
    vision: np.ndarray        # (N,2048) pooled
    frames: list[np.ndarray]  # per-utterance (T,512), for streaming replay
    frame_times: list[np.ndarray]
    has_vision: np.ndarray    # (N,) bool
    labels: np.ndarray
    dialogue_id: np.ndarray
    utterance_id: np.ndarray
    duration_s: np.ndarray
    speakers: np.ndarray
    texts: np.ndarray

    def __len__(self) -> int:
        return len(self.uids)

    @property
    def class_counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=N_EMO).astype(float)


def _load_clip(split: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    path = FEATURES / f"clip_{split}.npz"
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=False)
    uids, feats, times, off = z["uids"], z["feats"], z["times"], z["offsets"]
    return {
        str(u): (feats[off[i] : off[i + 1]].astype(np.float32),
                 times[off[i] : off[i + 1]])
        for i, u in enumerate(uids)
    }


def load_split(split: str) -> SplitFeatures:
    z = np.load(FEATURES / f"text_{split}.npz", allow_pickle=False)
    uids = [str(u) for u in z["uids"]]
    clip = _load_clip(split)

    frames, times, pooled, has = [], [], [], []
    for u in uids:
        f, t = clip.get(u, (np.zeros((0, 512), np.float32), np.zeros((0,), np.float32)))
        frames.append(f)
        times.append(t)
        pooled.append(pool_frames(f))
        has.append(len(f) > 0)

    return SplitFeatures(
        uids=np.array(uids),
        text_ctx=z["ctx"].astype(np.float32),
        text_solo=z["solo"].astype(np.float32),
        vision=np.stack(pooled).astype(np.float32) if pooled
        else np.zeros((0, VIS_POOL_DIM), np.float32),
        frames=frames,
        frame_times=times,
        has_vision=np.array(has, dtype=bool),
        labels=z["labels"],
        dialogue_id=z["dialogue_id"],
        utterance_id=z["utterance_id"],
        duration_s=z["duration_s"],
        speakers=z["speakers"],
        texts=z["texts"],
    )


def load_all() -> dict[str, SplitFeatures]:
    """Load whatever splits are cached.

    `make eval` only needs dev and test. Train features are ~110 MB and are not
    required to reproduce any reported number, so a repo can ship without them
    and the evaluation still runs end to end -- the class prior the majority
    baseline needs is already stored inside every head checkpoint.
    """
    out = {}
    for s in ("train", "dev", "test"):
        if (FEATURES / f"text_{s}.npz").exists():
            out[s] = load_split(s)
    missing = {"dev", "test"} - set(out)
    if missing:
        raise SystemExit(
            f"missing required features for {sorted(missing)}. "
            f"Run `make features` (streams MELD) to regenerate them.")
    return out

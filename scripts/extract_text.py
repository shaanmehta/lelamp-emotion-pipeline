"""Cache frozen RoBERTa features for every utterance, plus prefix features on
the test split for the accuracy-vs-coverage curve."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lelamp.config import FEATURES, DEVICE  # noqa: E402
from lelamp.data.meld import context_block, load_all  # noqa: E402
from lelamp.models.text import TextEncoder, prefix_text  # noqa: E402

PREFIX_FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BATCH = 128


def encode_all(enc: TextEncoder, texts: list[str]) -> np.ndarray:
    out = []
    for i in range(0, len(texts), BATCH):
        out.append(enc.encode(texts[i : i + BATCH]))
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


def main() -> int:
    import torch
    # Keep out of the way of the video decode pool if that is running too.
    torch.set_num_threads(2)
    enc = TextEncoder(device=DEVICE)
    print(f"[text] roberta-base loaded: {enc.n_params:,} params on {DEVICE}", flush=True)
    splits = load_all()
    t0 = time.time()

    for name, rows in splits.items():
        uids = [u.uid for u in rows]
        # Two variants so the ablation "does dialogue context help?" is real:
        # with prior turns, and utterance in isolation.
        ctx = encode_all(enc, [context_block(u) for u in rows])
        solo = encode_all(enc, [f"{u.speaker}: {u.text}" for u in rows])
        labels = np.array([u.label for u in rows], dtype=np.int64)
        dia = np.array([u.dialogue_id for u in rows], dtype=np.int64)
        utt = np.array([u.utterance_id for u in rows], dtype=np.int64)
        dur = np.array([u.duration_s for u in rows], dtype=np.float32)
        np.savez_compressed(
            FEATURES / f"text_{name}.npz",
            uids=np.array(uids), ctx=ctx.astype(np.float16),
            solo=solo.astype(np.float16), labels=labels,
            dialogue_id=dia, utterance_id=utt, duration_s=dur,
            speakers=np.array([u.speaker for u in rows]),
            texts=np.array([u.text for u in rows]),
        )
        print(f"[text] {name}: {len(rows)} utts -> text_{name}.npz "
              f"({time.time()-t0:.0f}s)", flush=True)

    # Prefix features for dev and test only. Dev is what the commit-vs-wait
    # policy is fitted on; test is what it is reported on. Doing this for train
    # would cost ~40 minutes and buy nothing.
    for split in ("dev", "test"):
        rows = splits[split]
        stack = []
        for f in PREFIX_FRACS:
            texts = []
            for u in rows:
                head = " </s> ".join(u.context)
                cur = f"{u.speaker}: {prefix_text(u.text, f)}"
                texts.append(f"{head} </s> {cur}" if head else cur)
            stack.append(encode_all(enc, texts).astype(np.float16))
            print(f"[text] {split} prefix {f:.1f} done ({time.time()-t0:.0f}s)",
                  flush=True)
        np.savez_compressed(
            FEATURES / f"text_{split}_prefix.npz",
            uids=np.array([u.uid for u in rows]),
            fracs=np.array(PREFIX_FRACS, dtype=np.float32),
            ctx=np.stack(stack),                   # (F, N, 768)
            labels=np.array([u.label for u in rows], dtype=np.int64),
        )
    print(f"[text] all done in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

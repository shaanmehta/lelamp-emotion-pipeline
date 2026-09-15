"""Add `solo` (no-context) prefix features to the cached prefix files.

The coverage curve has to be computed with the SAME head that is reported in the
ablation table. Once the fused head started consuming both the solo and the
context encodings, the prefix cache needed both too. Split out as its own script
rather than folded into extract_text.py so it can be run incrementally.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lelamp.config import DEVICE, FEATURES  # noqa: E402
from lelamp.data.meld import load_split  # noqa: E402
from lelamp.models.text import TextEncoder, prefix_text  # noqa: E402

BATCH = 128


def main() -> int:
    import torch
    torch.set_num_threads(2)
    enc = TextEncoder(device=DEVICE)
    t0 = time.time()
    for split in ("dev", "test"):
        path = FEATURES / f"text_{split}_prefix.npz"
        z = dict(np.load(path, allow_pickle=False))
        if "solo" in z:
            print(f"[solo] {split}: already present, skipping")
            continue
        rows = {u.uid: u for u in load_split(split)}
        uids = [str(u) for u in z["uids"]]
        stack = []
        for f in z["fracs"]:
            texts = [f"{rows[u].speaker}: {prefix_text(rows[u].text, float(f))}"
                     for u in uids]
            out = [enc.encode(texts[i:i + BATCH]) for i in range(0, len(texts), BATCH)]
            stack.append(np.concatenate(out).astype(np.float16))
            print(f"[solo] {split} {float(f):.1f} ({time.time()-t0:.0f}s)", flush=True)
        z["solo"] = np.stack(stack)
        np.savez_compressed(path, **z)
        print(f"[solo] {split}: wrote {path.name} "
              f"({path.stat().st_size/1e6:.1f} MB)", flush=True)
    print(f"[solo] done in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

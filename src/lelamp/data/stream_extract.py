"""Streams MELD.Raw.tar.gz over HTTP and turns it into CLIP features in one pass.

Why this exists: the tarball is 10.9GB and the laptop I built this on had about
12GB free, most of which the Python environment and model weights wanted.
Putting the archive on disk wasn't an option, and neither was extracting it.

So it never gets stored. I pull the gzip stream, walk the nested tars in memory,
decode only the clips I need, run them through the frozen CLIP tower, and throw
the bytes away. Peak disk cost is a couple hundred MB of features. Peak memory
is a handful of decoded frames.

The HTTP stream resumes by byte offset. Gzip is decoded incrementally over the
raw byte stream, so a Range re-request splices in without the decoder noticing.
A 20-minute download that dies at minute 18 is otherwise a very expensive way to
learn about TCP.
"""
from __future__ import annotations

import argparse
import faulthandler
import os
import random
import signal
import ssl
import sys
import tarfile
import threading
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi
import numpy as np

import torch

from ..config import CLIPS, FEATURES, MAX_FRAMES, MELD_RAW_URL
from ..models.vision import CLIP_DIM, VisionTower
from .meld import SPLIT_TARS, load_all
from .video import decode_frames

TAR2SPLIT = {v: k for k, v in SPLIT_TARS.items()}

# `kill -USR1 <pid>` dumps every thread's stack. Cheap to wire up, and the
# difference between "I think it deadlocked" and knowing where.
faulthandler.register(signal.SIGUSR1, all_threads=True)


def _rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        return -1.0


class ResumableStream:
    """Read-only byte stream over HTTP that survives connection drops."""

    # Long-lived connections to the CDN degrade over a multi-GB transfer. A
    # fresh Range request from the current offset restores full throughput and
    # costs one round trip, so we recycle the connection periodically rather
    # than only on error.
    RECONNECT_EVERY = 768 * 1024 * 1024

    def __init__(self, url: str, max_retries: int = 12):
        self.url, self.max_retries, self.pos = url, max_retries, 0
        self.last_recycle = 0
        # Homebrew Python ships no CA bundle; be explicit rather than disabling
        # verification, which is the tempting and wrong fix.
        self.ctx = ssl.create_default_context(cafile=certifi.where())
        self.resp = self._open(0)
        self.resumes = 0

    def _open(self, offset: int):
        req = urllib.request.Request(self.url)
        if offset:
            req.add_header("Range", f"bytes={offset}-")
        return urllib.request.urlopen(req, timeout=30, context=self.ctx)

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = 1 << 20
        out = bytearray()
        while len(out) < n:
            if self.pos - self.last_recycle >= self.RECONNECT_EVERY:
                self.last_recycle = self.pos
                try:
                    self.resp.close()
                except Exception:
                    pass
                self.resp = self._open(self.pos)
                print(f"  [stream] recycled connection at "
                      f"{self.pos/1e9:.2f} GB", flush=True)
            try:
                chunk = self.resp.read(n - len(out))
            except Exception:
                chunk = b""
                if self.resumes >= self.max_retries:
                    raise
                self.resumes += 1
                print(f"  [stream] reconnecting at byte {self.pos} "
                      f"(retry {self.resumes})", flush=True)
                time.sleep(2 * self.resumes)
                try:
                    self.resp.close()
                except Exception:
                    pass
                self.resp = self._open(self.pos)
                continue
            if not chunk:
                break  # genuine EOF
            out += chunk
            self.pos += len(chunk)
        return bytes(out)

    def close(self):
        try:
            self.resp.close()
        except Exception:
            pass


def _targets(train_cap: int, seed: int) -> tuple[dict[str, set[str]], dict]:
    """Which clips to decode. dev/test in full; train capped (see README)."""
    splits = load_all()
    want: dict[str, set[str]] = {}
    for name, rows in splits.items():
        uids = [u.uid for u in rows]
        if name == "train" and 0 < train_cap < len(uids):
            rng = random.Random(seed)
            # Sample at the DIALOGUE level, not the utterance level: context and
            # belief smoothing both operate per dialogue, so half a dialogue is
            # worth much less than a whole one.
            by_dia: dict[int, list[str]] = {}
            for u in rows:
                by_dia.setdefault(u.dialogue_id, []).append(u.uid)
            dia_ids = sorted(by_dia)
            rng.shuffle(dia_ids)
            picked: list[str] = []
            for d in dia_ids:
                if len(picked) >= train_cap:
                    break
                picked.extend(by_dia[d])
            uids = picked
        want[name] = set(uids)
    return want, splits


def _save(split: str, store: dict[str, tuple[np.ndarray, np.ndarray]]) -> Path:
    """Ragged (n_frames varies) features -> one flat npz with offsets."""
    uids = sorted(store)
    feats = np.concatenate([store[u][0] for u in uids]) if uids else np.zeros((0, CLIP_DIM), np.float16)
    times = np.concatenate([store[u][1] for u in uids]) if uids else np.zeros((0,), np.float32)
    counts = np.array([len(store[u][0]) for u in uids], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    out = FEATURES / f"clip_{split}.npz"
    np.savez_compressed(out, uids=np.array(uids), feats=feats.astype(np.float16),
                        times=times, offsets=offsets)
    print(f"  [save] {split}: {len(uids)} clips, {len(feats)} frames -> "
          f"{out.name} ({out.stat().st_size/1e6:.1f} MB)", flush=True)
    return out


def run(train_cap: int, seed: int, vendor_uids: set[str], limit_bytes: int | None,
        device: str) -> None:
    want, _ = _targets(train_cap, seed)
    total_want = sum(len(v) for v in want.values())
    print(f"[extract] targeting {total_want} clips "
          f"({ {k: len(v) for k, v in want.items()} })", flush=True)

    # Torch must not also try to use every core; the decode pool owns them.
    torch.set_num_threads(2)
    print(f"[extract] pid={os.getpid()}  (kill -USR1 for a stack dump)", flush=True)
    tower = VisionTower(device=device)
    print(f"[extract] CLIP vision tower loaded: {tower.n_params:,} params "
          f"on {device}", flush=True)

    store: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        s: {} for s in want
    }
    pending: deque = deque()
    batch_frames: list[np.ndarray] = []
    batch_meta: list[tuple[str, str, int, np.ndarray]] = []
    t0, done, skipped = time.time(), 0, 0
    mark = {"t": t0, "done": 0, "pos": 0}

    flushes = [0]

    def flush_batch():
        nonlocal batch_frames, batch_meta
        if not batch_frames:
            return
        emb = tower.encode(np.concatenate(batch_frames))
        i = 0
        for split, uid, n, times in batch_meta:
            store[split][uid] = (emb[i : i + n].astype(np.float16), times)
            i += n
        batch_frames, batch_meta = [], []
        flushes[0] += 1
        # Metal's caching allocator grows without bound across thousands of
        # small encodes. Left alone it pushes a 16 GB machine into swap and the
        # whole pipeline goes to ~2% CPU while the system pages -- which is
        # exactly how the first run of this script died.
        if flushes[0] % 25 == 0 and device == "mps":
            torch.mps.empty_cache()

    def collect(fut):
        nonlocal done
        split, uid, frames, times = fut.result()
        done += 1
        if len(frames):
            batch_frames.append(frames)
            batch_meta.append((split, uid, len(frames), times))
        if sum(len(f) for f in batch_frames) >= 32:
            flush_batch()
        if done % 200 == 0:
            now, el = time.time(), time.time() - t0
            dt = max(now - mark["t"], 1e-6)
            inst = (done - mark["done"]) / dt          # instantaneous, not mean:
            mbps = (stream.pos - mark["pos"]) / 1e6 / dt  # a mean hides a stall
            mark.update(t=now, done=done, pos=stream.pos)
            print(f"  [extract] {done}/{total_want} decoded  {inst:.1f} clip/s  "
                  f"net={stream.pos/1e9:.2f}GB ({mbps:.1f}MB/s now)  "
                  f"rss={_rss_mb():.0f}MB  "
                  f"eta={(total_want-done)/max(inst,1e-6)/60:.0f}m", flush=True)

    # Watchdog: if `done` stops advancing the run is wedged, and a silent hang
    # is far more expensive than a loud crash 25 minutes into a download.
    progress = {"done": 0, "t": time.time()}

    def watchdog():
        while not progress.get("stop"):
            time.sleep(20)
            if progress.get("stop"):
                return
            if done == progress["done"] and time.time() - progress["t"] > 120:
                print(f"[extract] WATCHDOG: no progress for 120s at {done} "
                      f"clips -- dumping stacks and aborting", flush=True)
                faulthandler.dump_traceback()
                os._exit(3)
            if done != progress["done"]:
                progress.update(done=done, t=time.time())

    threading.Thread(target=watchdog, daemon=True).start()

    stream = ResumableStream(MELD_RAW_URL)
    pool = ThreadPoolExecutor(max_workers=5)
    try:
        outer = tarfile.open(fileobj=stream, mode="r|gz")
        for member in outer:
            if not member.name.endswith(".tar.gz"):
                continue
            split = TAR2SPLIT.get(Path(member.name).name)
            if split is None:
                continue
            print(f"[extract] entering {member.name} "
                  f"({member.size/1e9:.2f} GB) -> split '{split}'", flush=True)
            inner_fh = outer.extractfile(member)
            inner = tarfile.open(fileobj=inner_fh, mode="r|gz")
            for clip in inner:
                if not clip.name.endswith(".mp4"):
                    continue
                uid = Path(clip.name).stem
                if uid not in want[split]:
                    skipped += 1
                    continue
                data = inner.extractfile(clip).read()
                # MELD numbers dialogues from 0 WITHIN each split, so
                # "dia237_utt3" exists in train and in test and they are
                # different clips. The demo quotes test labels, so only vendor
                # from test.
                if split == "test" and uid in vendor_uids:
                    (CLIPS / f"{uid}.mp4").write_bytes(data)
                pending.append(pool.submit(
                    lambda d=data, s=split, u=uid: (s, u, *decode_frames(d, MAX_FRAMES))
                ))
                while len(pending) >= 16:
                    collect(pending.popleft())
                if limit_bytes and stream.pos > limit_bytes:
                    raise KeyboardInterrupt("byte limit reached (probe mode)")
            inner.close()
            # Checkpoint immediately: test.tar.gz is last in the archive, so a
            # failure 25 minutes in must not also cost us the train features.
            while pending:
                collect(pending.popleft())
            flush_batch()
            if store[split]:
                _save(split, store[split])
    except KeyboardInterrupt as e:
        print(f"[extract] stopping early: {e}", flush=True)
    finally:
        while pending:
            collect(pending.popleft())
        flush_batch()
        pool.shutdown(wait=True)
        stream.close()

    progress["stop"] = True
    el = time.time() - t0
    print(f"[extract] done in {el/60:.1f} min  "
          f"({done} decoded, {skipped} skipped, {stream.resumes} reconnects)",
          flush=True)
    for split in ("train", "dev", "test"):
        if store[split] and not (FEATURES / f"clip_{split}.npz").exists():
            _save(split, store[split])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-cap", type=int, default=3500,
                    help="max train utterances to decode (0 = all 9989)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-mb", type=int, default=0,
                    help="stop after N MB of stream (smoke test)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--vendor", default="",
                    help="comma-separated uids to also save as mp4 in assets/clips")
    a = ap.parse_args(argv)
    from ..config import DEVICE
    run(a.train_cap, a.seed,
        {v for v in a.vendor.split(",") if v},
        a.probe_mb * 1_000_000 or None,
        a.device or DEVICE)
    return 0


if __name__ == "__main__":
    sys.exit(main())

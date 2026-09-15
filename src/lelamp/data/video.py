"""In-memory mp4 to CLIP-ready frames. No temp files, no ffmpeg binary needed —
PyAV bundles its own.
"""
from __future__ import annotations

import io

import av
import numpy as np

from ..models.vision import center_crop, crop_size

av.logging.set_level(av.logging.PANIC)  # MELD clips emit a lot of benign warnings


def decode_frames(
    data: bytes, max_frames: int = 8, size: int = 224
) -> tuple[np.ndarray, np.ndarray]:
    """Decode evenly-spaced frames from mp4 bytes.

    Single decode pass. We pick target frame indices up front from container
    metadata and only pay the (expensive) scale+colour-convert on frames we keep,
    so memory stays at a few hundred KB even with several decode threads running.

    Returns (frames uint8 [N,size,size,3] RGB, times seconds [N]).
    """
    empty = (np.zeros((0, size, size, 3), np.uint8), np.zeros((0,), np.float32))
    try:
        container = av.open(io.BytesIO(data))
    except Exception:
        return empty
    try:
        if not container.streams.video:
            return empty
        vs = container.streams.video[0]
        # Single-threaded decode ON PURPOSE. thread_type="AUTO" makes ffmpeg
        # spawn frame/slice threads *inside each container*; with a pool of
        # workers that becomes ~50 decode threads on an 8-core machine, the main
        # thread is starved, and the whole pipeline wedges at ~1% CPU. We get
        # our parallelism ACROSS clips instead, where it composes.
        vs.thread_type = "NONE"
        vs.codec_context.thread_count = 1

        n = vs.frames or 0
        if n <= 0:  # metadata missing: estimate from duration x frame rate
            dur = float(vs.duration * vs.time_base) if vs.duration else 0.0
            rate = float(vs.average_rate or 24)
            n = max(1, int(dur * rate))

        k = min(max_frames, n)
        targets = {int((i + 0.5) * n / k) for i in range(k)}

        frames, times, last = [], [], None
        for idx, frame in enumerate(container.decode(video=0)):
            last = (idx, frame)
            if idx not in targets:
                continue
            w, h = crop_size(frame.width, frame.height, size)
            arr = frame.reformat(width=w, height=h, format="rgb24").to_ndarray()
            frames.append(center_crop(arr, size))
            times.append(float(frame.pts * vs.time_base) if frame.pts is not None else 0.0)
            if len(frames) >= max_frames:
                break

        # Frame-count metadata can overestimate; make sure short clips still
        # yield at least one frame rather than silently returning nothing.
        if not frames and last is not None:
            frame = last[1]
            w, h = crop_size(frame.width, frame.height, size)
            arr = frame.reformat(width=w, height=h, format="rgb24").to_ndarray()
            frames.append(center_crop(arr, size))
            times.append(0.0)

        if not frames:
            return empty
        t = np.asarray(times, dtype=np.float32)
        t = t - t.min() if len(t) else t
        return np.stack(frames), t
    except Exception:
        return empty
    finally:
        container.close()

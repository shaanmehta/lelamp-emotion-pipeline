"""The only learned components we train ourselves: two small MLP heads on top of
frozen encoders.

Deliberate choice: we do NOT fine-tune RoBERTa or CLIP. Fine-tuning a large
backbone to claim a fine-tune is the anti-pattern here, and frozen features mean
every ablation retrains in seconds on CPU -- which is what makes the honest
modality comparison affordable at all.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..config import N_EMO
from .text import TEXT_DIM
from .vision import CLIP_DIM

VIS_POOL_DIM = CLIP_DIM * 4  # mean | max | std | (last - first)


def pool_frames(feats: np.ndarray) -> np.ndarray:
    """(T,512) per-frame CLIP embeddings -> (2048,) utterance descriptor.

    mean/max/std capture the scene; (last - first) captures CHANGE, which is the
    part that carries expression onset rather than who happens to be on camera.
    """
    if feats is None or len(feats) == 0:
        return np.zeros(VIS_POOL_DIM, dtype=np.float32)
    f = feats.astype(np.float32)
    mean, mx = f.mean(0), f.max(0)
    std = f.std(0) if len(f) > 1 else np.zeros_like(mean)
    delta = (f[-1] - f[0]) if len(f) > 1 else np.zeros_like(mean)
    return np.concatenate([mean, mx, std, delta])


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FusionHead(nn.Module):
    """One class covers text-only, vision-only and fused so the ablation is a
    config change, not three code paths that could diverge."""

    def __init__(self, use_text: bool = True, use_vision: bool = True,
                 hidden: int = 512, p: float = 0.3, n_text_fields: int = 1):
        super().__init__()
        assert use_text or use_vision
        self.use_text, self.use_vision = use_text, use_vision
        self.n_text_fields = n_text_fields if use_text else 0
        in_dim = (TEXT_DIM * self.n_text_fields) + (VIS_POOL_DIM if use_vision else 0)
        self.in_dim = in_dim
        self.head = MLPHead(in_dim, N_EMO, hidden, p)
        # Temperature is a single learned parameter fitted on dev, held fixed at
        # inference. It is in the ledger (one parameter, and we count it).
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, text=None, vision=None):
        parts = []
        if self.use_text:
            parts.append(text)
        if self.use_vision:
            parts.append(vision)
        return self.head(torch.cat(parts, dim=-1))

    @torch.inference_mode()
    def probs(self, text=None, vision=None, calibrated: bool = True):
        logits = self(text, vision)
        if calibrated:
            logits = logits / self.temperature.clamp_min(1e-3)
        return torch.softmax(logits, dim=-1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters()) + 1  # + temperature


class GatedFusionHead(nn.Module):
    """Late fusion with a learned per-utterance gate on the vision logits.

        logits = text_logits + g * vision_logits,   g = sigmoid(w . [t, v])

    Why not plain concatenation: the vision descriptor is 2048-d and carries
    weak signal on MELD (vision-only barely clears the majority baseline), while
    the text descriptor is 1536-d and carries most of it. Concatenating lets the
    head overfit the noisy half, and measurably does -- concat fusion scores
    BELOW its own text-only baseline.

    This form cannot: driving g to 0 recovers the text-only model exactly, so
    the optimiser has a free escape hatch wherever vision does not help. The
    gate is also worth reporting in its own right -- its mean value is a direct,
    interpretable readout of how much the model thinks the camera is worth.

    Shares the FusionHead interface so every downstream consumer (calibration,
    ablations, pipeline, streaming) is unchanged.
    """

    def __init__(self, n_text_fields: int = 2, hidden: int = 512,
                 p: float = 0.3):
        super().__init__()
        self.use_text = self.use_vision = True
        self.n_text_fields = n_text_fields
        t_dim = TEXT_DIM * n_text_fields
        self.in_dim = t_dim + VIS_POOL_DIM
        self.text_head = MLPHead(t_dim, N_EMO, hidden, p)
        self.vision_head = MLPHead(VIS_POOL_DIM, N_EMO, hidden // 2, p)
        self.gate = nn.Sequential(
            nn.LayerNorm(self.in_dim), nn.Linear(self.in_dim, 64), nn.GELU(),
            nn.Linear(64, 1),
        )
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, text=None, vision=None):
        g = torch.sigmoid(self.gate(torch.cat([text, vision], dim=-1)))
        return self.text_head(text) + g * self.vision_head(vision)

    @torch.inference_mode()
    def gate_value(self, text, vision):
        return torch.sigmoid(self.gate(torch.cat([text, vision], dim=-1)))

    @torch.inference_mode()
    def probs(self, text=None, vision=None, calibrated: bool = True):
        logits = self(text, vision)
        if calibrated:
            logits = logits / self.temperature.clamp_min(1e-3)
        return torch.softmax(logits, dim=-1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters()) + 1


class FastAffectHead(nn.Module):
    """Tier-1 reflexive head: ONE CLIP frame -> (valence, arousal, salience).

    It deliberately does not emit a 7-class emotion. Vision-only 7-way accuracy
    on MELD is barely above the majority baseline, so committing the lamp to a
    category at 200 ms would manufacture exactly the flicker the belief tracker
    exists to remove. Arousal is far more visually legible than category, and
    arousal is what the reflexive motion actually consumes (amplitude, speed).
    """

    def __init__(self, hidden: int = 256, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(CLIP_DIM),
            nn.Linear(CLIP_DIM, hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        y = self.net(x)
        # valence, arousal in [-1,1]; salience ("is anything happening") in [0,1]
        return torch.cat([torch.tanh(y[..., :2]), torch.sigmoid(y[..., 2:])], dim=-1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

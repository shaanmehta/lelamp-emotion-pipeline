"""CLIP ViT-B/32 vision tower, plus a fast tensor preprocessing path.

I load CLIPVisionModelWithProjection, never CLIPModel. The 63.4M-parameter CLIP
text tower is never instantiated, so it never needs to enter the ledger.
ledger.py asserts this rather than taking my word for it.
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import DEVICE, VISION_MODEL

# OpenAI CLIP normalisation constants.
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
CLIP_DIM = 512


def crop_size(w: int, h: int, target: int = 224) -> tuple[int, int]:
    """Resize-shortest-side-to-target dimensions, matching CLIP's preprocessing."""
    if w <= h:
        return target, max(target, round(h * target / w))
    return max(target, round(w * target / h)), target


def center_crop(arr: np.ndarray, target: int = 224) -> np.ndarray:
    h, w = arr.shape[:2]
    top, left = (h - target) // 2, (w - target) // 2
    return arr[top : top + target, left : left + target]


def normalize(frames_u8: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """(N,224,224,3) uint8 RGB -> (N,3,224,224) normalised float tensor.

    The uint8 array is uploaded and converted ON the target device. Doing the
    float32 conversion in numpy first allocates ~3x the batch size in host
    temporaries per call (42 MB -> 126 MB for a 32-frame batch), which is pure
    waste when the tensor is headed for the GPU anyway.
    """
    x = torch.from_numpy(np.ascontiguousarray(frames_u8)).to(device)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.as_tensor(CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.as_tensor(CLIP_STD, device=x.device).view(1, 3, 1, 1)
    return x.sub_(mean).div_(std)


class VisionTower:
    """Frozen CLIP image encoder. Returns L2-normalised 512-d projected embeddings."""

    def __init__(self, device: str = DEVICE):
        import logging

        from transformers import CLIPVisionModelWithProjection

        # The load report lists every CLIP *text* tower weight as UNEXPECTED.
        # That is the point -- it is proof the text tower is not loaded -- but it
        # is 40 lines of noise on every run, so log it once, quietly.
        logging.getLogger("transformers").setLevel(logging.ERROR)

        self.device = device
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            VISION_MODEL, dtype=torch.float32
        ).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def encode(self, frames_u8: np.ndarray) -> np.ndarray:
        """(N,224,224,3) uint8 -> (N,512) float32, L2-normalised."""
        if len(frames_u8) == 0:
            return np.zeros((0, CLIP_DIM), dtype=np.float32)
        x = normalize(frames_u8, self.device)
        emb = self.model(pixel_values=x).image_embeds
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return emb.float().cpu().numpy()

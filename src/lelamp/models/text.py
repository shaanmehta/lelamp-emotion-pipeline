"""Frozen RoBERTa-base utterance encoder.

Mean-pooled over the attention mask rather than <s>: RoBERTa's CLS token is only
meaningful after fine-tuning, and we deliberately do not fine-tune (see README,
"What we did not do"). Mean pooling is the right read-out for a frozen encoder.
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import DEVICE, TEXT_MODEL

TEXT_DIM = 768
MAX_LEN = 128


class TextEncoder:
    def __init__(self, device: str = DEVICE):
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
        self.model = AutoModel.from_pretrained(TEXT_MODEL, dtype=torch.float32)
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, TEXT_DIM), dtype=np.float32)
        b = self.tok(texts, padding=True, truncation=True, max_length=MAX_LEN,
                     return_tensors="pt").to(self.device)
        h = self.model(**b).last_hidden_state              # (B,T,768)
        m = b["attention_mask"].unsqueeze(-1).float()
        emb = (h * m).sum(1) / m.sum(1).clamp_min(1.0)     # masked mean pool
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return emb.float().cpu().numpy()


def prefix_text(text: str, frac: float) -> str:
    """Whitespace-token prefix, simulating an incremental transcript.

    This is the honest approximation available in the Text+Vision track: with no
    audio there is no VAD, so we cannot do real endpointing. What we CAN model is
    the decision problem -- how much of the utterance you have heard vs. whether
    to commit -- and that is what the accuracy-vs-coverage curve measures.
    """
    words = text.split()
    if not words:
        return text
    k = max(1, int(round(len(words) * frac)))
    return " ".join(words[:k])

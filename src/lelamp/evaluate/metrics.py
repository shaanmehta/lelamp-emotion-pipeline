"""Shared evaluation helpers. Every number in the README goes through here."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from ..config import EMOTIONS, HEADS, N_EMO
from ..data.features import SplitFeatures
from ..models.heads import FusionHead, GatedFusionHead


def load_head(name: str, device: str = "cpu") -> tuple[FusionHead, dict]:
    ck = torch.load(HEADS / f"{name}.pt", map_location=device, weights_only=False)
    if ck.get("gated"):
        m = GatedFusionHead(n_text_fields=ck["n_text_fields"])
    else:
        m = FusionHead(use_text=ck["use_text"], use_vision=ck["use_vision"],
                       n_text_fields=ck.get("n_text_fields", 1))
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck


@torch.inference_mode()
def predict_probs(model: FusionHead, ck: dict, fx: SplitFeatures,
                  calibrated: bool = True) -> np.ndarray:
    t = (torch.from_numpy(
            np.concatenate([fx.__dict__[f] for f in ck["text_fields"]], 1))
         if ck["use_text"] else None)
    v = torch.from_numpy(fx.vision) if ck["use_vision"] else None
    return model.probs(t, v, calibrated=calibrated).cpu().numpy()


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "weighted_f1": round(float(f1_score(y, pred, average="weighted", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "accuracy": round(float((y == pred).mean()), 4),
    }


def per_class_f1(y: np.ndarray, pred: np.ndarray) -> dict:
    f = f1_score(y, pred, average=None, labels=range(N_EMO), zero_division=0)
    return {e: round(float(x), 4) for e, x in zip(EMOTIONS, f)}


def confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y, pred, labels=range(N_EMO))


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> tuple[float, list]:
    """Expected calibration error on the max-probability (confidence) axis."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0,
                         "conf": 0.0, "acc": 0.0})
            continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        total += m.mean() * abs(a - c)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "conf": c, "acc": a})
    return float(total), bins


def risk_coverage(probs: np.ndarray, y: np.ndarray, n: int = 40) -> list[dict]:
    """Selective performance: if the lamp abstains below a confidence threshold,
    how good is it on what it does commit to, and how often does it commit?"""
    conf, pred = probs.max(1), probs.argmax(1)
    out = []
    for tau in np.linspace(0.0, conf.max() * 0.99, n):
        m = conf >= tau
        if m.sum() < 20:
            break
        out.append({
            "tau": round(float(tau), 4),
            "coverage": round(float(m.mean()), 4),
            "weighted_f1": round(float(f1_score(y[m], pred[m], average="weighted",
                                                zero_division=0)), 4),
            "accuracy": round(float((y[m] == pred[m]).mean()), 4),
        })
    return out


def majority_baseline(train_counts: np.ndarray, test: SplitFeatures) -> dict:
    """Majority class is taken from the TRAIN prior, never from test."""
    maj = int(np.asarray(train_counts).argmax())
    pred = np.full(len(test.labels), maj)
    s = scores(test.labels, pred)
    s["predicts"] = EMOTIONS[maj]
    s["test_base_rate"] = round(float((test.labels == maj).mean()), 4)
    return s

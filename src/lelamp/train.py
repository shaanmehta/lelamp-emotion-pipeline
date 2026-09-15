"""Trains the small heads on frozen features.

Every modality configuration goes through the same code, on the same rows, with
the same schedule. That's the point: if text-only and fused differ, the
difference is the modality and not some incidental change in how they were
trained.

Runs on CPU in well under a minute per config, which is what made it affordable
to report all the ablations instead of just the one that won.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from .config import EMOTIONS, HEADS, N_EMO, VA_COORDS
from .data.features import SplitFeatures, load_all
from .models.heads import FastAffectHead, FusionHead, GatedFusionHead
from .runtime.policy import prior_offset

# name -> (tuple of text feature fields, use_vision)
#
# `text_ctx` mean-pools RoBERTa over [3 prior turns] + [current utterance].
# That turns out to be actively HARMFUL on frozen features -- the context tokens
# dilute the current utterance in the mean -- so the fused config keeps the two
# encodings SEPARATE and concatenates them, which preserves the utterance signal
# while still letting the head use context. `fused_ctx_only` is kept so the
# ablation shows what that choice is worth.
CONFIGS = {
    "text_solo": (("text_solo",), False),
    "text_ctx": (("text_ctx",), False),
    "text_pair": (("text_solo", "text_ctx"), False),
    "vision": ((), True),
    "fused_ctx_only": (("text_ctx",), True),
    "fused_concat": (("text_solo", "text_ctx"), True),
    # headline config: gated late fusion (see models/heads.GatedFusionHead)
    "fused": (("text_solo", "text_ctx"), True),
}
GATED = {"fused"}


def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)


def tensors(fx: SplitFeatures, text_fields: tuple, use_vision: bool):
    t = (torch.from_numpy(np.concatenate([fx.__dict__[f] for f in text_fields], 1))
         if text_fields else None)
    v = torch.from_numpy(fx.vision) if use_vision else None
    y = torch.from_numpy(fx.labels)
    return t, v, y


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
    }


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Single-parameter temperature scaling (Guo et al. 2017), fitted on dev.

    Test is never touched. One scalar, and it IS counted in the ledger.
    """
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_t.exp().clamp_min(1e-3), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def train_one(name: str, feats: dict[str, SplitFeatures], epochs: int, seed: int,
              class_weight: bool, device: str = "cpu") -> dict:
    text_fields, use_vision = CONFIGS[name]
    set_seed(seed)
    if name in GATED:
        model = GatedFusionHead(n_text_fields=len(text_fields)).to(device)
    else:
        model = FusionHead(use_text=bool(text_fields), use_vision=use_vision,
                           n_text_fields=len(text_fields)).to(device)

    # Heads train on CPU by default: they are ~1.5M parameters on cached
    # features, so a GPU round trip costs more than it saves, and CPU keeps the
    # ablations bit-for-bit reproducible.
    tr_t, tr_v, tr_y = tensors(feats["train"], text_fields, use_vision)
    dv_t, dv_v, dv_y = tensors(feats["dev"], text_fields, use_vision)

    w = None
    if class_weight:
        counts = feats["train"].class_counts
        w = torch.tensor((counts.sum() / (N_EMO * counts)) ** 0.5, dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # Modality dropout, only for genuinely multimodal configs. Without it the
    # fused head leans on the vision branch harder than the vision branch can
    # support on MELD and lands BELOW its own text-only baseline. Zeroing a
    # modality at random forces a working single-modality path to exist inside
    # the fused head, which is also what you want on a robot -- the camera view
    # is frequently useless (user out of frame, dark room) and the model should
    # degrade to text rather than degrade to noise.
    multimodal = tr_t is not None and tr_v is not None
    p_drop_v, p_drop_t = (0.25, 0.10) if multimodal else (0.0, 0.0)

    n = len(tr_y)
    best, best_state, patience, t0 = -1.0, None, 0, time.time()
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 128):
            idx = perm[i : i + 128]
            bt = tr_t[idx] if tr_t is not None else None
            bv = tr_v[idx] if tr_v is not None else None
            if multimodal:
                dv = (torch.rand(len(idx), 1, generator=g) < p_drop_v).float()
                dt = (torch.rand(len(idx), 1, generator=g) < p_drop_t).float()
                dt = dt * (1 - dv)          # never drop both at once
                bv = bv * (1 - dv)
                bt = bt * (1 - dt)
            opt.zero_grad()
            loss = lossf(model(bt, bv), tr_y[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            dv_logits = model(dv_t, dv_v)
        f1 = f1_score(dv_y.numpy(), dv_logits.argmax(1).numpy(),
                      average="weighted", zero_division=0)
        if f1 > best:
            best, patience = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 6:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        dv_logits = model(dv_t, dv_v)
    temp = fit_temperature(dv_logits, dv_y)
    model.temperature.fill_(temp)

    out = {"config": name, "dev_weighted_f1": float(best), "temperature": temp,
           "epochs_run": ep + 1, "train_seconds": round(time.time() - t0, 1),
           "head_params": model.n_params,
           "modality_dropout": {"vision": p_drop_v, "text": p_drop_t}}
    if name in GATED:
        with torch.no_grad():
            out["mean_gate_dev"] = round(
                float(model.gate_value(dv_t, dv_v).mean()), 4)
    torch.save({"state_dict": model.state_dict(),
                "use_text": bool(text_fields), "text_fields": list(text_fields),
                "n_text_fields": len(text_fields),
                "use_vision": use_vision, "temperature": temp,
                "gated": name in GATED,
                "class_counts": feats["train"].class_counts.tolist(),
                "va_offset": prior_offset(feats["train"].class_counts).tolist(),
                "meta": out}, HEADS / f"{name}.pt")
    return out


def train_fast_head(feats: dict[str, SplitFeatures], epochs: int, seed: int,
                    device: str = "cpu") -> dict:
    """Tier-1 head: single CLIP frame -> (valence, arousal, salience).

    Targets are the DETERMINISTIC VA projection of the gold label, plus a binary
    "is this non-neutral" salience flag. Note the label noise this introduces:
    MELD labels the utterance, and we apply that label to every frame in it. Some
    frames genuinely carry no expression. Reported as a limitation.
    """
    set_seed(seed)
    va = np.array([VA_COORDS[e] for e in EMOTIONS], dtype=np.float32)
    off = prior_offset(feats["train"].class_counts).astype(np.float32)
    from .config import VA_SCALE
    scale = np.array(VA_SCALE, dtype=np.float32)

    def build(fx: SplitFeatures):
        X, Y = [], []
        for f, lab in zip(fx.frames, fx.labels):
            if len(f) == 0:
                continue
            tgt = np.concatenate([
                np.clip((va[lab] - off) / scale, -1, 1),
                [0.0 if EMOTIONS[lab] == "neutral" else 1.0],
            ])
            X.append(f)
            Y.append(np.repeat(tgt[None], len(f), axis=0))
        if not X:
            return torch.zeros(0, 512), torch.zeros(0, 3)
        return (torch.from_numpy(np.concatenate(X)).float(),
                torch.from_numpy(np.concatenate(Y)).float())

    trX, trY = build(feats["train"])
    dvX, dvY = build(feats["dev"])
    model = FastAffectHead().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    mse, bce = nn.MSELoss(), nn.BCELoss()

    best, best_state, t0 = 1e9, None, time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(trX))
        for i in range(0, len(trX), 512):
            idx = perm[i : i + 512]
            opt.zero_grad()
            p = model(trX[idx])
            loss = mse(p[:, :2], trY[idx, :2]) + 0.5 * bce(
                p[:, 2].clamp(1e-6, 1 - 1e-6), trY[idx, 2])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = model(dvX)
            dl = float(mse(p[:, :2], dvY[:, :2]) + 0.5 * bce(
                p[:, 2].clamp(1e-6, 1 - 1e-6), dvY[:, 2]))
        if dl < best:
            best, best_state = dl, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    with torch.no_grad():
        p = model(dvX).numpy()
    out = {
        "config": "fast_affect",
        "dev_loss": round(best, 4),
        # Correlation is the number that matters: does one frame tell us anything
        # about arousal at all? Honest answer goes in the README.
        "dev_r_valence": round(float(np.corrcoef(p[:, 0], dvY[:, 0].numpy())[0, 1]), 4),
        "dev_r_arousal": round(float(np.corrcoef(p[:, 1], dvY[:, 1].numpy())[0, 1]), 4),
        "dev_salience_auc": round(_auc(dvY[:, 2].numpy(), p[:, 2]), 4),
        "head_params": model.n_params,
        "train_seconds": round(time.time() - t0, 1),
        "n_train_frames": int(len(trX)),
    }
    torch.save({"state_dict": model.state_dict(), "meta": out},
               HEADS / "fast_affect.pt")
    return out


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--class-weight", action="store_true",
                    help="sqrt-inverse-frequency CE weights (raises macro-F1, "
                         "lowers weighted-F1; off by default for comparability)")
    a = ap.parse_args(argv)

    feats = load_all()
    for s, fx in feats.items():
        print(f"[data] {s}: {len(fx)} utts, vision on "
              f"{int(fx.has_vision.sum())} ({fx.has_vision.mean()*100:.1f}%)")

    results = {}
    for name in CONFIGS:
        r = train_one(name, feats, a.epochs, a.seed, a.class_weight)
        results[name] = r
        print(f"[train] {name:10s} dev_wF1={r['dev_weighted_f1']:.4f} "
              f"T={r['temperature']:.3f} ep={r['epochs_run']} "
              f"({r['train_seconds']}s, {r['head_params']:,} params)")

    r = train_fast_head(feats, max(a.epochs, 15), a.seed)
    results["fast_affect"] = r
    print(f"[train] fast_affect r_val={r['dev_r_valence']:.3f} "
          f"r_aro={r['dev_r_arousal']:.3f} sal_auc={r['dev_salience_auc']:.3f} "
          f"({r['head_params']:,} params)")

    (HEADS / "train_summary.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

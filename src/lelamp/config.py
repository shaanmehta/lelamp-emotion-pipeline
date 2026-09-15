"""Single source of truth for paths, model ids and tunable constants.

Everything the README quotes as a number should be reachable from here, so the
write-up and the code cannot drift apart.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CSV_DIR = DATA / "meld_csv"
ARTIFACTS = ROOT / "artifacts"
# Overridable so a dry run can exercise the full train/eval path against
# synthetic features without touching the real cache.
FEATURES = Path(os.environ.get("LELAMP_FEATURES", ARTIFACTS / "features"))
HEADS = Path(os.environ.get("LELAMP_HEADS", ARTIFACTS / "heads"))
FIGURES = ARTIFACTS / "figures"
RUNS = ARTIFACTS / "runs"
ASSETS = ROOT / "assets"
CLIPS = ASSETS / "clips"

for _p in (DATA, CSV_DIR, ARTIFACTS, FEATURES, HEADS, FIGURES, RUNS, ASSETS, CLIPS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- models ----
# Vision tower only. We instantiate CLIPVisionModel, NOT CLIPModel, so the
# 63.4M-parameter text tower is never loaded. ledger.py asserts this at runtime.
VISION_MODEL = "openai/clip-vit-base-patch32"
TEXT_MODEL = "roberta-base"
# 4-bit MLX build; ~0.9 GB download. Pipeline degrades to the template responder
# when mlx_lm is unavailable, so this is never a hard dependency.
LLM_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

MELD_RAW_URL = (
    "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz"
)

# ---------------------------------------------------------------- labels ----
EMOTIONS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
E2I = {e: i for i, e in enumerate(EMOTIONS)}
N_EMO = len(EMOTIONS)

# Valence/arousal is a DETERMINISTIC PROJECTION of the emotion posterior onto
# fixed circumplex coordinates -- MELD carries no VA labels, so we do not and
# cannot learn these. Stated plainly in the README; do not describe the system
# as "predicting valence and arousal".
VA_COORDS = {
    "neutral":  (0.00, -0.10),
    "joy":      (0.80,  0.50),
    "sadness":  (-0.70, -0.45),
    "anger":    (-0.75,  0.70),
    "surprise": (0.15,  0.80),
    "fear":     (-0.65,  0.75),
    "disgust":  (-0.60,  0.45),
}
# After de-biasing by the class prior the projection no longer spans [-1,1],
# so we rescale by the observed half-range. Without this, realistic posteriors
# (max-prob ~0.7, not 1.0) land so close to the origin that every behaviour
# region collapses to "idle". Verified by tests/test_policy.py.
VA_SCALE = (0.75, 0.65)

# ------------------------------------------------------------- streaming ----
FPS_SAMPLE = 5.0          # frames/second pulled from the video stream
MAX_FRAMES = 8            # cap per utterance (MELD median duration ~2.5 s)
REFLEX_TICK_MS = 200      # reflexive tier cadence
CONTEXT_TURNS = 3         # prior turns fed to the text encoder

# ------------------------------------------------------------- behaviour ----
EMA_ALPHA = 0.6           # belief EMA on the probability simplex
HYSTERESIS_MARGIN = 0.15  # new emotion must beat incumbent by this to switch
MIN_DWELL_MS = 700        # ...and the incumbent must have been held this long
TAU_ABSTAIN = 0.45        # calibrated max-prob below this => UNCERTAIN

# --------------------------------------------------------------- budgets ----
# Justified in README section "Latency budget". p95 targets, milliseconds.
BUDGET_REFLEX_MS = 300
BUDGET_TTFT_MS = 800

DEVICE = os.environ.get("LELAMP_DEVICE", "mps")

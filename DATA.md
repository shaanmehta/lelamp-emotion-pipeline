# Where the data and models come from

## MELD

Poria et al., *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in
Conversations*, ACL 2019. https://affective-meld.github.io/

- **Transcripts and labels** are committed under `data/meld_csv/` (1.5MB) so the
  demo runs with no network. `src/lelamp/data/meld.py:ensure_csv` re-downloads
  them from the official repo (`declare-lab/MELD`) if they go missing.
- **Video** is streamed from `MELD.Raw.tar.gz` (10.9GB) and never written to
  disk. See `src/lelamp/data/stream_extract.py`.
- **`assets/clips/`** holds 19 MELD test clips from dialogues 237 and 60. They're
  in the repo so `make demo` works without a 10.9GB download. Included for
  reproducing the demo only, under the dataset's research-use terms, credited to
  the authors above.
- **`artifacts/features/`** holds frozen CLIP and RoBERTa embeddings. Derived
  features, not redistributable media.

MELD is licensed for research use. See the upstream repository for terms.

## Pretrained models

All downloaded at runtime from Hugging Face. None of them are fine-tuned here.

| Model | Licence | What it does |
|---|---|---|
| `openai/clip-vit-base-patch32` | MIT | frame encoder (vision tower only) |
| `roberta-base` | MIT | utterance and context encoder |
| `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | Apache 2.0 | response text |

No remote inference anywhere in the pipeline. Everything runs on the machine.

## What I trained

Only the small heads, on cached frozen features:

- `FastAffectHead` (0.13M) — tier-1 valence/arousal/salience
- `GatedFusionHead` (1.56M) — tier-2 emotion posterior
- `FusionHead` variants (0.4M–1.06M) — the ablation configurations and the
  evidence heads
- One temperature scalar per head, fitted on dev

Total trained: about 1.6M parameters. Everything else is frozen and pretrained.

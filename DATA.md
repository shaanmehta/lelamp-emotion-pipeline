# Data provenance

## MELD
Poria et al., *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in
Conversations*, ACL 2019. https://affective-meld.github.io/

- **Transcripts / labels** are fetched from the official repo
  (`declare-lab/MELD`) into `data/meld_csv/` on first use.
- **Video** is streamed from `MELD.Raw.tar.gz` (10.9 GB) and never stored. See
  `src/lelamp/data/stream_extract.py`.
- **`assets/clips/`** contains a small number of MELD test clips (dialogues 237
  and 60), vendored so that `make demo` runs offline without a 10.9 GB download.
  They are included for reproducibility of the demo only, under the dataset's
  research-use terms, and are credited to the authors above.
- **`artifacts/features/`** holds frozen CLIP and RoBERTa embeddings. These are
  derived features, not redistributable media.

MELD is licensed for research purposes; see the upstream repository for terms.

## Pretrained models (all downloaded at runtime from Hugging Face)
| Model | Licence |
|---|---|
| `openai/clip-vit-base-patch32` | MIT |
| `roberta-base` | MIT |
| `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | Apache 2.0 |

No remote inference is used anywhere in the pipeline.

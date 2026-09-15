# LeLamp emotion pipeline

Text + Vision on MELD. Runs locally. 1.81B of the 6B parameter budget.

## What I built and why it looks like this

MELD is a classification benchmark. LeLamp is a lamp. It has no face and no
hands, so everything it can express comes out through motion, light, and timing.
I built the system around that instead of around the leaderboard.

Four decisions follow from it:

1. The output is a belief over time, not a per-clip argmax. A lamp that flips
   joy to anger to neutral in 300ms looks broken. So I smooth the state and I
   report switches per minute, not just F1.
2. There are two latency deadlines, not one. The body has to react fast. The
   voice can take a beat. I never let the body's deadline depend on the language
   model.
3. Low confidence is a real output. The behaviour policy reads calibrated
   confidence, not argmax, and the lamp is allowed to say nothing.
4. MELD gives you whole utterances. Real speech doesn't arrive that way. So I
   measured accuracy against how much of the utterance has been heard, and
   learned a policy for when to stop waiting.

## Running it

```bash
make setup     # venv + deps, about 2 min
make demo      # one MELD dialogue streamed through the lamp
```

`make demo` needs no MELD download. The clips and cached features are in the
repo. On a new machine it does pull three checkpoints from Hugging Face the
first time (about 2.1GB: CLIP 0.6, RoBERTa 0.5, Qwen-1.5B-4bit 1.0). After that
it runs offline. Either way inference is local. There are no API calls anywhere.

The demo replays a real MELD test dialogue at wall-clock speed and draws the
lamp in your terminal. About 25s at `--speed 10`, about 4 min at 1x.

```bash
make grounding # same words, different video, different behaviour
make eval      # regenerates every number and figure below (~2 min)
make ledger    # parameter count
make test      # 25 tests
```

If you want to rebuild the features from the raw dataset:

```bash
make features  # streams MELD.Raw.tar.gz (10.9GB) without storing it, then
               # retrains every head. About 25 min.
make train     # just the heads, from cached features. CPU, under a minute.
```

### What's in the repo

| | size | needed for |
|---|---|---|
| `artifacts/features/` dev+test embeddings | 143.5 MB | `make eval` |
| `artifacts/heads/` 7 trained heads | 30.6 MB | everything |
| `assets/clips/` 19 MELD test clips | 19.2 MB | `make demo`, `make grounding` |
| `data/meld_csv/` transcripts and labels | 1.5 MB | everything |
| `artifacts/figures/` 8 figures | 0.4 MB | this README |
| total | 196 MB | |

Train-split features are about 99MB and I left them out. Nothing in this README
needs them: the class prior the majority baseline uses is stored inside each
head checkpoint. I checked this rather than assuming it, by moving the train
caches aside and re-running `make eval`. Every figure came out the same.
`make features` rebuilds them if you want to retrain.

### The data path

MELD ships as one 10.9GB tarball. The laptop I built this on had about 12GB
free, and most of that was going to the Python environment and model weights.
So storing the archive was not an option, and neither was extracting it.

The extractor never stores it. It streams the gzip over HTTP, walks the nested
tars in memory, decodes each clip from a byte buffer, runs it through the frozen
CLIP tower, and throws the bytes away. Peak disk cost is about 110MB of
features. The HTTP stream resumes by byte offset, so a dropped connection 20
minutes in splices back in instead of starting over.

I'd rather have had the disk space, but the constraint produced a better
extractor than I would have written otherwise.

### What the demo shows

The terminal lamp is driven only by the `behavior` block of the emitted JSON. If
the schema were missing something a robot needs, the renderer couldn't draw it.
That was the point of writing it.

```
╭─────────────────────────── LeLamp — test dialogue 237 ───────────────────────╮
│     ___     tier       deliberative         t+3130 ms                        │
│    /###\    emotion    ANGER                                                 │
│    |   |    confidence ██████████████░░░░   0.75                             │
│     //      valence    ░░███████│░░░░░░░░   -0.80                            │
│   __|__     arousal    ░░░░░░░░░│█████░░░   +0.60                            │
│                                                                              │
│             intent     recoil               priority 3                       │
│             motion     lean_back / speaker  amp 0.72  spd 0.80               │
│             light      ■■■  hue 3°          sat 0.74  val 0.63               │
│             timing     hold 919 ms          decay 679 ms  expr 0.80          │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Underneath it you can watch the two tiers interleave. Reflexive events fire
while the person is still talking, then one commit at the end:

```
t=     0ms  reflex   v=-0.05 a=+0.18  attend
t=   333ms  reflex   v=-0.12 a=+0.31  attend
t=  1833ms  reflex   v=-0.48 a=+0.55  recoil        <- body has already moved
t=  3130ms  COMMIT   anger 0.75 -> recoil  (end of utterance)
            speech   llm ttft=812ms
```

That layout is illustrative; the values depend on the clip. Real measured
latencies are in the results section. Nothing in that block is a claim.

Every event gets appended to `artifacts/runs/*.jsonl`.

## Results

<!-- BEGIN GENERATED RESULTS -->

*Generated by `make eval` on 2026-09-15 19:36:46. Every number here comes out of `artifacts/results.json`.*

### Baselines and modality ablation, MELD test (2,610 utterances)

| config | weighted F1 | macro F1 | accuracy | ECE | head params |
|---|---|---|---|---|---|
| majority (`neutral`, 48.1% of test) | 0.3127 | 0.0928 | 0.4812 | — | 0 |
| vision only | 0.3490 | 0.1673 | 0.3778 | 0.0694 | 1,056,777 |
| text only, context mean-pooled with utterance | 0.4985 | 0.2962 | 0.5330 | 0.0285 | 398,857 |
| text only, utterance alone | 0.6225 | 0.4456 | 0.6467 | 0.0377 | 398,857 |
| text only, utterance ⊕ context (separate) | 0.6286 | 0.4328 | 0.6452 | 0.0264 | 793,609 |
| fused, context mean-pooled | 0.4815 | 0.2840 | 0.5161 | 0.0278 | 1,451,529 |
| fused, plain concatenation | 0.6039 | 0.4093 | 0.6142 | 0.0363 | 1,846,281 |
| **fused (utterance ⊕ context ⊕ vision, gated)** | 0.6175 | 0.4381 | 0.6310 | 0.0278 | 1,560,721 |

Published MELD weighted-F1 sits in the mid-60s, with fine-tuned backbones. My best configuration is `text_pair` at **0.6286**, and the multimodal head I ship is at **0.6175**, with every encoder frozen and about 1.6M trained parameters. Landing just under the published range is what not fine-tuning costs, and I think it's the right trade here: frozen features are the reason I could afford to run and report seven configurations instead of only the one that won. **Anything above about 0.67 would mean a leak**, which is why the leak check below exists.

**Fusion buys -0.0111 weighted F1** over the strongest text-only configuration (`text_pair`). I measured it against the best baseline rather than a convenient one. On the 30% of utterances where that text head is least certain (n=783), it's -0.0297.

**A result I didn't expect:** adding dialogue context the naive way *hurts*, by 0.1240 weighted F1 (0.6225 down to 0.4985). Mean-pooling frozen RoBERTa over [3 prior turns + current utterance] dilutes the utterance that actually carries the label. Encoding the two separately and concatenating gets it back (`text_pair` = 0.6286). Published MELD work gets large gains from context, but it fine-tunes. With frozen features, how you inject context matters more than whether you do.

| vision helps most | Δ F1 | | vision hurts | Δ F1 |
|---|---|---|---|---|
| disgust | +0.0800 | | anger | -0.0675 |
| fear | +0.0491 | | surprise | -0.0170 |
| joy | +0.0018 | |  |  |

Not every distinction matters equally to a lamp. Accuracy restricted to each confusable pair — neutral vs anger is the difference between `idle` and `recoil`:

| pair | text only | fused | Δ |
|---|---|---|---|
| neutral vs anger | 0.7651 | 0.7164 | -0.0487 |
| neutral vs surprise | 0.7866 | 0.7781 | -0.0085 |
| neutral vs joy | 0.7750 | 0.7636 | -0.0114 |
| neutral vs sadness | 0.7548 | 0.7336 | -0.0212 |


The fused head trains with **modality dropout**: vision zeroed on 25% of training samples, text on 10%, never both. It costs about 0.002 dev F1, so it isn't there for the metric. It's there because a lamp's camera is useless far more often than a benchmark suggests — user out of frame, back turned, dark room — and the model should fall back to text rather than to noise. I measured that instead of assuming it, by zeroing the vision features at inference:

| fused head, at inference | weighted F1 | macro F1 |
|---|---|---|
| with camera | 0.6175 | 0.4381 |
| **camera unavailable** | 0.6198 | 0.4451 |
| (reference: best text-only head) | 0.6286 | 0.4328 |

#### What I conclude about fusion

**On MELD, the vision branch as I built it does not pay for itself.** Fusion is -0.0111 weighted F1 against the best text-only head, and the clearest version of that is this: take the same fused head, zero its vision input at inference, and it scores **0.6198**, slightly better than running it with the camera (0.6175). The gate is open, mean around 0.6, so the model is using vision. It's just mildly hurting.

In the design phase I predicted vision would earn its place by fixing the neutral/anger confusion, since that's the distinction the lamp's behaviour leans on most. **That was wrong.** The pair table above shows neutral vs anger getting worse with vision. Where vision does help is `disgust` and `fear`, the two rarest classes, at 268 and 271 training examples each. My guess is that facial and postural cues survive where there isn't enough text to learn from, but at those class counts I wouldn't bet on it replicating.

So would I put this vision branch on a real lamp? **For classification, no.** It costs 87.8M parameters and about 13ms a frame to make the posterior slightly worse. The real case for the camera isn't in this table at all: it's the reflexive tier, which is the only part of the system that can move the lamp before the sentence ends, and which weighted-F1 can't measure by construction. That's a reason to keep the camera and a strong reason not to believe it improves classification.

![ablation](artifacts/figures/06_ablation.png)
![confusion](artifacts/figures/05_confusion.png)

#### Leak check

MELD's six main cast appear in every split, so a vision head can get credit for recognising an actor instead of an expression. They speak 83% of test utterances. A big main-cast/everyone-else gap on the vision-only config would be the tell:

| config | main cast wF1 | everyone else wF1 | gap |
|---|---|---|---|
| vision | 0.3357 | 0.4154 | -0.0797 |
| text_solo | 0.6105 | 0.6811 | -0.0706 |
| fused | 0.6032 | 0.6852 | -0.0820 |

**No leak, and the gap runs the other way** (-0.0797 for vision-only). The main cast are harder, not easier. That's what you'd expect if the model is reading expression rather than identity: the six leads carry the emotional range of the show, while background characters are mostly neutral one-liners. If vision-only had shown a big positive gap here, the vision number would have been actor recognition in disguise.

### Accuracy against how much has been heard

![coverage](artifacts/figures/01_coverage_curve.png)

Mean MELD utterance is **3133ms**. The fused model gets within 1 F1 point of its full-utterance score after **100%** of the utterance, about 3133ms.

Read this as a lower bound. The fusion head trained on whole utterances and I'm evaluating it on prefixes, so what this measures is how gracefully a full-utterance model degrades, not what a model built for streaming would do.

One detail with real consequences: mean confidence *falls* as more is heard (0.754 at 10% to 0.609 at 100%) while accuracy rises (0.4304 to 0.6186). The temperature was fitted on complete utterances, so the model is **overconfident on short prefixes**. Any commit rule that thresholds confidence inherits that, which is exactly what the next section runs into.

#### Commit vs wait (λ = 0.35)

| policy | mean reward | weighted F1 | mean heard |
|---|---|---|---|
| **learned (fitted-Q)** | 0.5288 | 0.4306 | 10% |
| fixed_conf_0.30 | 0.5294 | 0.4323 | 10% |
| fixed_conf_0.40 | 0.5248 | 0.4360 | 12% |
| fixed_conf_0.50 | 0.5207 | 0.4554 | 16% |
| fixed_conf_0.60 | 0.5065 | 0.4809 | 22% |
| fixed_step_1 | 0.4990 | 0.4550 | 20% |

λ is a product decision, not something to learn, so I swept it rather than quoting one value:

| λ | learned reward | heard | best fixed baseline | its reward | gain |
|---|---|---|---|---|---|
| 0.15 | 0.5317 | 40% | `fixed_conf_0.60` | 0.5339 | -0.0022 |
| 0.25 | 0.5260 | 13% | `fixed_conf_0.30` | 0.5297 | -0.0037 |
| 0.35 | 0.5288 | 10% | `fixed_conf_0.30` | 0.5294 | -0.0006 |
| 0.50 | 0.5291 | 10% | `fixed_conf_0.30` | 0.5291 | +0.0000 |

**The learned policy doesn't beat a well-chosen fixed threshold.** It matches it, within 0.0006 reward at every λ. That's a negative result and I'm reporting it as one. The reason is in the section above: confidence is miscalibrated on short prefixes, so the main feature the policy has to reason with is unreliable exactly where the decision gets interesting. Fitting the temperature per prefix length is the first thing I'd try, and it's cheap. I'm keeping this in rather than quietly dropping it, because the extension was optional and the finding is more useful than the feature would have been.

To be precise about the setup: I observe the reward for every prefix, not just the one I chose, so this is offline policy learning with full-information feedback. That's strictly easier than the online bandit a deployed robot faces. Calling it a contextual bandit would be overstating it.

### Calibration and abstention

![reliability](artifacts/figures/02_reliability.png)
![risk-coverage](artifacts/figures/03_risk_coverage.png)

Temperature scaling (T = 1.298, fitted on dev, never on test) takes ECE from **0.0593** to **0.0278**. At the abstention threshold I ship, τ = 0.45, the lamp declines to commit on **26.1%** of utterances and scores **0.7059** on the rest.

This matters beyond the metric. Calibrated confidence is what the commit-vs-wait policy thresholds against, so an overconfident head commits too early and the policy pays for it.

### Emotional inertia

![smoothing](artifacts/figures/04_smoothing.png)

| | weighted F1 | switches / minute of speech |
|---|---|---|
| raw argmax | 0.6175 | 9.3 |
| smoothed (`ema(a=0.6)+hysteresis(m=0.15)+dwell(700ms)`) | 0.5381 | 3.9 |

Where to sit on this is a product decision, not an accuracy optimisation. How twitchy a lamp is allowed to look is a question for a human study, not for weighted-F1. So here's the frontier, with the point I shipped marked:

| EMA α | hysteresis margin | weighted F1 | switches/min | |
|---|---|---|---|---|
| 1.0 | 0.00 | 0.6094 | 9.1 |  |
| 1.0 | 0.10 | 0.6112 | 8.0 |  |
| 0.8 | 0.00 | 0.6064 | 8.0 |  |
| 1.0 | 0.15 | 0.6095 | 7.6 |  |
| 0.8 | 0.10 | 0.5930 | 6.5 |  |
| 1.0 | 0.25 | 0.5862 | 6.4 |  |
| 0.6 | 0.00 | 0.5855 | 6.3 |  |
| 0.8 | 0.15 | 0.5803 | 5.8 |  |
| 0.6 | 0.10 | 0.5577 | 4.6 |  |
| 0.8 | 0.25 | 0.5542 | 4.5 |  |
| 0.4 | 0.00 | 0.5436 | 4.3 |  |
| 0.6 | 0.15 | 0.5381 | 3.9 | **shipped** |
| 0.6 | 0.25 | 0.5072 | 2.6 |  |
| 0.4 | 0.10 | 0.4970 | 2.5 |  |
| 0.4 | 0.15 | 0.4778 | 1.9 |  |
| 0.4 | 0.25 | 0.4474 | 1.2 |  |

**58% fewer emotion switches** for -0.0794 weighted F1. The F1 column is the one to distrust. MELD emotions are sticky inside a dialogue, so smoothing can gain F1 for reasons that have nothing to do with the lamp looking better. The switch-rate column is what the behaviour layer actually pays.

### Is the fusion doing anything, or is it decoration?

![counterfactual](artifacts/figures/07_counterfactual.png)

Hold the transcript fixed, swap in a different utterance's video, and measure how far the posterior moves. Across all 2610 test utterances, not one hand-picked example, because with 2,610 of them you can always find one that looks good.

| swap | mean total variation | emotion flips | behaviour intent changes |
|---|---|---|---|
| video (same words) | 0.1478 | 17.0% | 36.6% |
| words (same video) | 0.5040 | 63.0% | 72.2% |

Text moves the posterior **3.4x** more than video does. That's text dominance, and MELD is a text-dominant dataset, so I'm not going to pretend otherwise. But the video channel isn't inert: swapping it changes the behaviour the lamp performs on 36.6% of utterances, and that's the level the robot actually operates at.

`make grounding` shows a concrete pair: same transcript, two different videos, two different states, two different spoken lines.

### Measured latency

*macOS-14.5-arm64-arm-64bit, device `mps`, responder Qwen2.5-1.5B-Instruct-4bit (MLX). Measured over 40 runs after warm-up.*

![latency](artifacts/figures/08_latency.png)

| stage | p50 (ms) | p95 (ms) |
|---|---|---|
| mp4 decode (8 frames) | 169.9 | 341.4 |
| CLIP, 1 frame | 18.9 | 21.0 |
| CLIP, 8 frames | 65.3 | 69.7 |
| **TIER 1 total (frame → behaviour)** | 19.9 | 23.4 |
| RoBERTa encode | 16.0 | 18.1 |
| temporal pooling | 0.1 | 0.1 |
| 3 fusion heads | 2.3 | 3.1 |
| belief update | 0.0 | 0.1 |
| behaviour policy | 0.1 | 0.1 |
| **TIER 2 total (→ COMMIT_STATE)** | 3.0 | 3.8 |
| **time to first spoken token** | 552.9 | 560.4 |

- **Reflexive budget 300ms: met**, p95 = 23ms, about 13x margin. The lamp moves well inside the window where a motion still reads as a reaction.
- **Verbal budget 800ms to first token: met**, p95 = 560ms.
- **`mp4 decode` is deliberately not inside the reflexive total.** It's an artefact of replaying a dataset: on a robot the camera hands you decoded frames and there's no container to demux. I left it in the table rather than dropping it, because it is a real cost of this harness and hiding it would make the 23ms look cheaper than the measurement was.
- Perception real-time factor p95 = **0.260**, which has to be well under 1 to consume a live stream. Peak RSS **1678MB**.

### Parameter ledger

*Counted by `make ledger`, which sums `.numel()` over the modules that actually get instantiated. Nothing here is typed in by hand.*

| component | role | parameters |
|---|---|---|
| `openai/clip-vit-base-patch32` | vision tower (frames -> 512d) | 87,849,216 |
| `roberta-base` | utterance+context encoder | 124,645,632 |
| `FastAffectHead` | tier-1 reflexive v/a/salience | 133,123 |
| `GatedFusionHead[fused]` | tier-2 emotion posterior (+1 temperature) | 1,560,720 |
| `FusionHead[text_pair]` | evidence: text-only margin | 793,608 |
| `FusionHead[vision]` | evidence: vision-only margin / disagreement | 1,056,776 |
| `BeliefTracker` | EMA + hysteresis + dwell | 0 |
| `BehaviorPolicy` | state -> intent/light/motion | 0 |
| `TemplateResponder` | zero-parameter fallback speech | 0 |
| `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | response generation (loaded and counted) | 1,591,950,848 |
| **TOTAL** | | **1,807,989,923** |
| budget | | 6,000,000,000 |
| **headroom** | | **4,192,010,077 (69.9%)** |

Two things I made sure this doesn't hide:

1. **CLIP's text tower is never loaded.** I instantiate `CLIPVisionModelWithProjection`, not `CLIPModel`, so the 63.4M text tower doesn't exist at runtime. `ledger.py` asserts this rather than claiming it — swap the class back and the ledger refuses to run.
2. **The 4-bit LLM is counted at full logical width.** MLX packs eight 4-bit weights into each `uint32`, so naively summing array sizes gives about 241M for Qwen2.5-1.5B. That would understate the ledger by 1.3 *billion* parameters, in my favour. I unpack it. The 1,591,950,848 reported is slightly above the dense checkpoint total because quantisation scales get counted too.

And the one I don't get to skip. MELD hands me gold transcripts, so no ASR, VAD, or TTS sits in my path. That's a gift from the task, not a property of a lamp. Priced in:

| additional component | role | parameters |
|---|---|---|
| `openai/whisper-small` | streaming ASR | 244,000,000 |
| `silero-vad v4` | voice activity / endpointing | 1,100,000 |
| `piper en_US-lessac-medium` | TTS | 20,000,000 |
| `SCRFD-500m` | face detect + active speaker gating | 600,000 |
| **PROJECTED TOTAL** | | **2,073,689,923** (65.4% headroom) |

<!-- END GENERATED RESULTS -->

## Architecture

```
                     ┌──────────────── STREAM SOURCE ────────────────┐
                     │ MELD clip replayed at wall-clock rate:        │
                     │   frames at their true decode timestamps      │
                     │   words at duration/n_words                   │
                     └───────┬───────────────────────────┬───────────┘
                             │ frame(t)                  │ word(t)
                             ▼                           ▼
 ══ TIER 1 · REFLEXIVE ── runs every 200 ms, DURING the utterance ══
   ┌──────────────────────────┐
   │ CLIP ViT-B/32 vision twr │  1 frame → 512-d              (87.8 M)
   └───────────┬──────────────┘
               ▼
   ┌──────────────────────────┐   emits valence + arousal + salience.
   │ FastAffectHead (MLP)     │   Not a 7-class emotion. See below.
   └───────────┬──────────────┘                               (0.13 M)
               ▼
        PARTIAL_STATE  ──────────────────►  body can move now
               │
 ══ TIER 2 · DELIBERATIVE ── at commit ═════════════════════════════
               │
   ┌───────────┴───────────┐     ┌────────────────────────────┐
   │ RoBERTa-base (124.6 M)│     │ frame buffer → temporal    │
   │ TWO encodings, kept   │     │ pool: mean‖max‖std‖Δ       │
   │ separate on purpose:  │     │ → 2048-d  (reuses tier-1   │
   │  [utt]        → 768-d │     │ encoder, 0 new params)     │
   │  [3 prior][utt]→768-d │     └──────────────┬─────────────┘
   └───────────┬───────────┘                    │
               │  mean-pooling these together   │
               │  costs 12 F1 points — see      │
               │  Results                       │
               └──────────┬─────────────────────┘
                          ▼
            ┌────────────────────────┐     ┌──────────────────────────┐
            │ GatedFusionHead        │     │ text-only head  (0.79 M) │
            │  logits_text           │     │ vision-only head(1.06 M) │
            │  + g · logits_vision   │     │  → disagreement signal   │
            │  + temp. scaling       │     └──────────────────────────┘
            │              (1.56 M)  │
            │  g→0 recovers text-only│
            └───────────┬────────────┘
                        ▼
            ┌────────────────────────┐
            │ BeliefTracker          │  EMA on the simplex + margin
            │ (0 params)             │  hysteresis + minimum dwell
            └───────────┬────────────┘
                        ▼
            ┌────────────────────────┐
            │ BehaviorPolicy         │  reads (valence, arousal,
            │ (0 params, pure fn)    │  confidence), not argmax
            └───────────┬────────────┘
                        ▼
                COMMIT_STATE ────────────►  body, before any speech
                        │
                        ▼
            ┌────────────────────────┐
            │ Qwen2.5-1.5B-Instruct  │  streams tokens
            │ 4-bit MLX     (1.59 B) │
            └────────────────────────┘
              ↳ TemplateResponder (0 params) if the LLM isn't there
```

### Why these models and not the obvious ones

| Block | What I used | The obvious alternative | Why |
|---|---|---|---|
| Vision | CLIP ViT-B/32 vision tower, frozen | An AffectNet-style face CNN behind a face detector | The face route needs a detector, and MELD shots are wide and often have three to six people in them. Which face is talking is unsolved. CLIP encodes the whole frame, so I get the face when it's large plus posture, lighting, and shot type. Frozen means no training cost. The cost is that I never work out who is speaking. That's in the limits section. |
| Text | RoBERTa-base, frozen, mean-pooled | DeBERTa-v3-base (about 2 points better); or reuse the LLM's hidden states | RoBERTa is about 25ms on MPS and it's simple. Reusing the LLM would save 125M, but the classifier has to run before the LLM, and frozen decoder embeddings are worse than encoder mean-pooling for sentence classification. There's 70% headroom left, so DeBERTa is a one-line swap if the numbers need it. |
| Fusion | Gated late fusion: `logits_text + g·logits_vision` | Plain concat into an MLP; cross-attention; end-to-end fine-tuning | I built concat first. It scores below its own text-only baseline (0.604 vs 0.629), because putting 2048 noisy vision dims next to 1536 useful text dims invites the head to overfit the noisy half. The gated version can drive `g` to 0 and get the text-only model back exactly, so there's an escape hatch wherever vision doesn't help. Both are in the ablation table. Everything trains in seconds on CPU, which is the reason I could afford to report seven configurations instead of just the winner. |
| Responder | Qwen2.5-1.5B-Instruct, 4-bit, MLX | Qwen2.5-3B, which the budget allows | This is an M1, not an M1 Pro. 3B roughly doubles prefill and the verbal budget is already the tighter one. The ledger shows I could afford 3B, which I think is a better argument than using the biggest thing that fit. |
| Tier-1 output | valence + arousal + salience | 7-class emotion from a single frame | Vision-only 7-way on MELD is barely above majority. Committing the lamp to a category at 200ms would manufacture exactly the flicker the belief tracker exists to remove. Arousal is much more visible in a frame than category is, and arousal is what reflexive motion actually needs (amplitude, speed). |

## Latency budget

Two budgets, because the interaction has two deadlines.

**Reflexive, 300ms.** Gaps between turns in human conversation sit around 200ms.
Past roughly 500ms a response starts to mean something on its own — hesitation,
reluctance. A lamp movement that lands later than that stops reading as a
reaction to the thing that triggered it and starts reading as the lamp doing
something unrelated.

**Verbal, 800ms to first token, measured from the end of the utterance.** A lamp
gets to take a beat. Unlike a voice assistant, a short pause reads as character.
Past about a second the user starts repeating themselves.

The design consequence that matters: `COMMIT_STATE` is emitted before the
responder runs, so the body's deadline is never tied to the language model's.

## The output contract

Schema version 1.0. Both event types use one envelope so a consumer only needs
one parser. A robotics engineer only needs `behavior.{intent, priority, hold_ms,
decay_ms, expressivity}`. Posteriors and evidence are diagnostic. I split it that
way so perception can change how it computes emotion without breaking motion.

Eight intents, closed set: `idle, attend, acknowledge, celebrate, soothe,
recoil, startle, confused_tilt`. They're picked from (valence, arousal,
confidence), not from `argmax(probs)`. So a posterior split 0.45 anger / 0.40
disgust produces something coherent in between instead of snapping to whichever
won by 0.05, and low confidence routes to `attend` or `confused_tilt` with
`speech.allowed = false`.

```jsonc
{
  "schema_version": "1.0",
  "utterance_id": "dia237_utt6", "dialogue_id": "dia237",
  "tier": "deliberative",              // "reflexive" | "deliberative"
  "t_emit_ms": 3130.0,
  "coverage": { "text_frac": 1.0, "frames_seen": 8, "audio": false },

  "affect": {
    "emotion": "anger",                // one of MELD's 7; null on reflexive events
    "probs": { "neutral": 0.09, "joy": 0.02, "sadness": 0.05, "anger": 0.71,
               "surprise": 0.08, "fear": 0.02, "disgust": 0.03 },
    "confidence": 0.71, "calibrated": true,
    "valence": -0.80, "arousal": 0.60,
    "uncertain": false                 // confidence < tau_abstain
  },

  // everything a motion engineer needs is in this block and only this block
  "behavior": {
    "intent": "recoil", "priority": 3,
    "hold_ms": 919, "decay_ms": 679,
    "expressivity": 0.80,              // global gain, scales with confidence
    "light":  { "hue_deg": 3.4, "sat": 0.74, "value": 0.63, "pulse_hz": 0.0 },
    "motion": { "posture": "lean_back", "gaze": "speaker",
                "amplitude": 0.72, "speed": 0.80 }
  },

  "evidence": {                        // diagnostic: lets you audit fusion live
    "dominant_modality": "text", "text_margin": 0.42, "vision_margin": 0.11,
    "modality_agreement": 0.83,        // 1 - total variation(text, vision)
    "visual_cue": "high visual change across the utterance"
  },

  "speech": { "text": "Okay - that landed hard. I'm listening.",
              "style": "measured", "onset_delay_ms": 183, "allowed": true },

  "belief": {                          // what the smoothing did, so you can see it
    "smoothing": "ema(a=0.6)+hysteresis(m=0.15)+dwell(700ms)",
    "raw_emotion": "anger", "prev_emotion": "neutral",
    "switched": true, "dwell_ms": 3400, "suppressed_switches": 1
  },

  "latency": { "pool_ms": 0.5, "heads_ms": 1.9, "belief_ms": 0.1,
               "policy_ms": 0.2, "ttft_ms": 812.0, "responder": "llm" }
}
```

A reflexive event is the same envelope with `tier: "reflexive"`,
`affect.emotion: null`, `affect.probs: null`, `speech: null`. One parser handles
both, and anything that only cares about moving reads `behavior` either way.

`tests/test_policy.py` asserts that each emotion lands in the behaviour region I
intended. Those tests caught two real bugs while I was building: hue
interpolating through green on the way from calm to angry (interpolating degrees
linearly takes the long way round the colour wheel), and the prior de-bias
flattening every arousal region into `idle`.

## Decisions you'll probably ask about

**Why Text+Vision and not Text+Audio?** Audio is the better product bet and I'd
ship it first on a real lamp. Prosody is there whenever someone speaks; vision
fails when the user is out of frame, backlit, or turned away. I picked vision
here for two reasons. MELD's audio is a studio mix with a laugh track in it, so
prosodic features off it transfer to a kitchen worse than coarse visual affect
does. And frames arrive from t=0 and keep arriving, which is what let me build a
reflexive tier that fires before the utterance ends without pulling VAD and ASR
into the parameter budget.

**Valence and arousal aren't predicted.** MELD has no VA labels. They're a
deterministic projection of the emotion posterior onto fixed circumplex
coordinates, de-biased by the training class prior so that "posterior equals the
prior" lands on (0,0) — no information should read as neutral, not as mildly
negative — then rescaled to span [-1,1]. Calling this "predicting valence and
arousal" would be wrong. It predicts emotion and projects it.

**There's no endpointing.** No audio means no VAD, so real endpointing isn't
available in this track and I didn't fake one. What I did build is the half that
drives behaviour: the commit-vs-wait decision over an incremental transcript.
The coverage curve and the learned commit policy are that half, measured.

**The tier-1 head trains on noisy labels.** MELD labels the utterance. I apply
that label to every frame in it. Some frames carry no expression at all. That
inflates tier-1's training loss and it's a real weakness.

**Smoothing might help F1 for the wrong reason.** MELD emotions are sticky
within a dialogue, so temporal smoothing can look like an accuracy win that's
really a dataset-structure win. That's why the headline smoothing number here is
switches per minute rather than F1, and why both are reported next to each other.

**`visual_cue` is not a face reading.** There's no face detector. It describes
how much the CLIP embedding moves across the utterance, which is an honest
description of what the visual branch actually contributes.

### Two things that cost me real time

**PyAV's `thread_type="AUTO"` doesn't compose with a worker pool.** Every open
container spawns its own ffmpeg frame and slice threads, so a pool of seven
decode workers turned into roughly fifty threads on an eight-core machine,
starved the main thread, and wedged the extractor at about 1% CPU after a few
hundred clips. Twice, silently. The fix is single-threaded decode per container
with parallelism across clips instead. The extractor now runs a watchdog that
dumps every thread's stack and aborts loudly rather than hanging, and
`kill -USR1 <pid>` dumps stacks on demand. That's what found it.

**Two processes using Metal at once reliably wedge one of them** on this
machine. The CLIP extractor and a RoBERTa job running together, and the second
one drops to about 2% CPU and never comes back. That's why `scripts/finish.sh`
is strictly sequential. Worth knowing before designing a multi-process
perception stack on Apple Silicon.

## Domain gap

MELD is acted American sitcom dialogue. Professional actors, studio lighting, a
laugh track, cuts mid-utterance, and six main characters who show up in every
split.

**What probably transfers.** Coarse text sentiment. The shape of the VA
projection. The belief smoothing and commit-vs-wait machinery, since those are
about decision dynamics and not about sitcoms. The behaviour contract.

**What probably doesn't.**

- Base rates. Neutral is about 48% of MELD test. In a kitchen it's far higher,
  because most of what a lamp hears is unremarkable. A model tuned on MELD's
  priors will over-react. The abstention threshold is the knob, and it would
  need refitting on real data.
- Acted intensity. Sitcom anger is performed at broad amplitude for a camera.
  Being annoyed at home is a flat voice and a small change in the face.
- Framing. MELD gives medium shots with several people and directorial cuts. A
  lamp gets one person, close, off-centre, often looking away, under whatever
  lighting the room has.
- Speaker identity as a shortcut. The same six actors are in train, dev, and
  test, so a vision head can get credit for recognising Ross instead of reading
  an expression. I checked for this explicitly (see `leak_check` in
  `artifacts/results.json`) by splitting test into main cast and everyone else.

**To close it** I'd collect a few hours of in-home, consented, wake-word-gated
interaction. One user, real room lighting, real mic distance, labelled for
whether the lamp should have reacted and how, rather than for a 7-way emotion
category. The label a robot needs isn't the label MELD gives you.

## What I finished and what I left out

**Finished.**

- Streaming feature extraction over the raw 10.9GB MELD archive without storing it.
- Two-tier runtime: reflexive vision path and deliberative fusion path, with the
  state event emitted before speech.
- Full structured output contract, versioned, with a terminal visualiser driven
  only by the `behavior` block.
- Modality ablation, baselines, per-class breakdown, calibration, abstention,
  belief smoothing, accuracy-vs-coverage curve, aggregate counterfactual
  grounding, measured per-stage latency, and a generated parameter ledger.
- A learned commit-vs-wait policy against fixed-threshold baselines.
- Local response generation with a zero-parameter fallback.

**Left out on purpose.**

- **A live camera and mic path.** I had no webcam available for this, and an
  untested capture path in a repo someone runs once is a liability. The stream
  interface in `runtime/sources.py` is three event types wide precisely so a
  capture source drops in. I didn't ship one I couldn't test.
- **Audio, and therefore the three-modality extension.** Adding it meant either
  a rushed third branch or cutting evidence from the core. I cut the feature,
  not the evidence.
- **Fine-tuning any backbone.** Frozen encoders are what made the ablation
  affordable. Every configuration retrains in seconds, so I could report the
  comparison instead of just the one that won.
- **TTS.** It would go in the ledger. The projection table prices it in.
- **Sampling and beam-search tuning on the responder.** Not where the
  interesting risk was.

## What I'd do with two more weeks and a real robot

1. **Change the label.** Stop predicting MELD's 7 categories and start
   predicting whether a behaviour was appropriate. Collect pairs of (context,
   candidate lamp behaviour) and label which one reads better. That's the target
   the product actually has, and it steps around the acted-sitcom domain gap
   entirely.
2. **Put audio in front.** VAD, streaming ASR, and a prosody branch, with the
   reflexive tier driven by prosody instead of frames. That's the version I'd
   ship. The ledger has room: projected 2.07B of 6B.
3. **Active-speaker gating.** A small face detector plus mouth-motion
   correlation, so the visual branch reads the person talking instead of the
   scene. Biggest single quality fix available to the vision path.
4. **Close the loop on the commit policy online.** What I learned here is an
   offline policy with full-information rewards, which is easier than the real
   problem. On a robot you only see the reward for the action you took, and the
   honest version is an online contextual bandit with a real interruption cost.
5. **Calibrate per prefix length.** Highest-value fix in the repo. Confidence is
   fitted on whole utterances and is overconfident on short prefixes, which is
   why the learned commit policy has nothing to work with. One temperature per
   coverage bucket is a few lines.
6. **Measure the thing that actually matters.** Latency and F1 are proxies. The
   real question is whether a person in the room reads the lamp as responsive.
   I'd run a small within-subject study: same transcript, reflexive tier on and
   off, ask which lamp seems to be listening.
7. **Prompt-cache the system prompt.** TTFT is dominated by prefill over a
   constant instruction block. Caching its KV buys enough headroom to move to a
   3B responder, which the budget already allows.

## Repo map

```
src/lelamp/
  config.py              paths, model ids, every tunable constant
  schema.py              the output contract (dataclasses -> JSON)
  ledger.py              parameter ledger, counted not quoted
  train.py               trains all heads + calibration, CPU, under a minute
  data/
    meld.py              CSV loading, causal dialogue context, split bookkeeping
    video.py             in-memory mp4 -> CLIP-ready frames, no ffmpeg binary
    stream_extract.py    streams the 10.9GB tarball without storing it
    features.py          joins cached text + vision features
  models/
    vision.py            CLIP ViT-B/32 vision tower, text tower never loaded
    text.py              frozen RoBERTa-base, masked mean pooling
    heads.py             FusionHead, GatedFusionHead, FastAffectHead
    responder.py         Qwen2.5-1.5B via MLX + zero-parameter template fallback
  runtime/
    sources.py           ReplayStream: frames and words arriving over time
    belief.py            EMA + margin hysteresis + minimum dwell
    policy.py            (valence, arousal, confidence) -> behaviour contract
    pipeline.py          the two-tier orchestrator
    visualizer.py        terminal lamp, driven only by `behavior`
  evaluate/
    metrics.py ablations.py streaming.py smoothing.py
    counterfactual.py latency.py plots.py run_all.py
scripts/
  demo.py                streaming demo + counterfactual demo
  extract_text.py        caches RoBERTa features including prefixes
  extract_solo_prefix.py adds no-context prefix features
  render_results.py      regenerates the Results section of this README
tests/                   25 tests: behaviour contract, belief, causality,
                         schema, head-loading invariants
```

## Summary

This is a small system built around one idea: a robot needs a belief about how
someone feels, held over time, with a confidence attached, and a behaviour
contract a motion engineer can build against without reading any of the ML code.

It doesn't beat published MELD numbers and it isn't trying to. Everything is
frozen and only about 1.6M parameters are trained. What it does instead is
report the things that decide whether a lamp feels alive: when the reaction
lands, how often the state flickers, whether the confidence means anything, how
much of a sentence you need before you can move, and whether the second modality
changes any behaviour at all.

Where the evidence is unflattering I put it in the README instead of a footnote.
The vision branch doesn't pay for itself on MELD — the fused head scores better
with its camera input zeroed. The learned commit policy only ties a fixed
threshold. Smoothing costs real F1. The vision branch can't tell which face is
speaking. The parts that did work — 22ms p95 on the reflexive path against a
300ms budget, 571ms p95 to first spoken token against 800ms, calibration that
makes abstention mean something, and 58% less state flicker — are measured on
the hardware named above, not estimated.

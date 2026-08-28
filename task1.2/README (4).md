# Task 1.2 — Image Caption Generator

**AI Team Training '27 — Task 6, Phase II**

An end-to-end, production-oriented Image Caption Generator: upload a photo, get an
automatically generated English caption. Built on the Flickr8k dataset, combining
computer vision (image feature extraction) with NLP (sequence generation), and
organized as a modular Python package rather than a single notebook — with a
deployable Gradio interface and a Docker image.

---

## 1. Project overview

The system takes an image and generates a natural-language description of it,
combining two components:

```
Image → EfficientNet-B3 (CNN, pretrained) → spatial image features
      → Bahdanau attention + LSTM decoder → generated caption
```

Unlike a simple "one feature vector per image" approach, the encoder keeps a
**spatial grid of features** so the decoder's attention mechanism can focus on
different regions of the image as it generates each word (e.g. look at the dog while
generating "dog", then the ball while generating "ball").

## 2. Dataset

- **Flickr8k**: ~8,091 images, each paired with 5 human-written reference captions.
- **Preprocessing**: captions lowercased and stripped of non-alphabetic characters;
  images resized to 300×300 and normalized with ImageNet statistics (required for
  any pretrained torchvision CNN).
- **Split**: 81% train / 9% validation / 10% test (810 test images), split **by unique
  image filename** — not by caption row — so all 5 captions of a given image always
  land in the same split. This avoids data leakage between splits.
- **Vocabulary**: word-level, built from training captions only, with a minimum
  frequency threshold (`min_freq=5`) to fold rare/unseen words into `<unk>` — a
  deliberate choice for a dataset with high per-image caption diversity relative to
  its size.

## 3. Architecture

### Encoder — EfficientNet-B3 (CNN)
- Pretrained on ImageNet, classification head removed; outputs a 7×7 spatial grid of
  1536-dimensional features (via `.features` + adaptive average pooling).
- **Partial fine-tuning**: early convolutional blocks are frozen (generic
  edge/texture/color features don't need relearning on only 8k images); later blocks
  are fine-tuned to adapt to the dataset's domain. This satisfies the "use transfer
  learning rather than training from scratch" requirement while still adapting
  higher-level features.

### Decoder — LSTM with Bahdanau (additive) attention
- At each decoding step, an attention module scores every one of the 49 spatial
  locations against the decoder's current hidden state, producing a weighted context
  vector — i.e. which part of the image is relevant to the word being generated
  right now.
- A learned **gate** (`f_beta`) scales the attention context per step, letting the
  model down-weight the image signal for function words (e.g. "a", "the") where
  visual grounding matters less — a technique from the *Show, Attend and Tell* paper.
- The decoder's initial hidden state is derived from the mean-pooled image features
  (instead of zeros), giving it a head start on "what the image is about."
- Implemented with `LSTMCell` unrolled manually one timestep at a time (rather than a
  single `LSTM` call), since attention must be recomputed against the image at every
  step.

### Loss regularization
- Standard cross-entropy loss (padding ignored), plus a **doubly-stochastic attention
  regularization** term that encourages the total attention paid to each image region
  (summed across all decoding steps) to be close to 1 — discouraging the model from
  fixating on a single salient region and ignoring the rest of the image.

## 4. Training process

- Adam optimizer (`lr=3e-4`), `ReduceLROnPlateau` scheduler, gradient clipping
  (`max_norm=5.0`), early stopping (`patience=4` on validation loss), checkpointing
  of the best model only.
- Only the decoder's parameters and the unfrozen tail of the encoder are optimized
  (`model.trainable_parameters()`), so frozen encoder blocks aren't wastefully
  tracked by the optimizer.
- Trained on a Kaggle GPU (Tesla T4). Training stopped early at **epoch 14** (best
  weights from **epoch 10**), after **107.3 minutes**.

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 4.1520 | 3.4995 |
| 4 | 2.8498 | 2.9396 |
| 8 | 2.3743 | 2.8229 |
| **10 (best)** | 2.2107 | **2.8081** |
| 14 (stopped) | 1.8756 | 2.8302 |

Train loss kept improving through epoch 14, while validation loss plateaued and
began rising slightly after epoch 10 — a mild overfitting pattern, expected given the
small dataset (8k images) relative to model capacity. Early stopping correctly
restored the epoch-10 weights.

## 5. Evaluation metrics

Evaluated on the **full test set (810 images)**, each generated caption scored
against **all 5** of its reference captions (not just one), using greedy decoding:

- **BLEU** — n-gram precision (1–4-gram) with a brevity penalty, computed
  corpus-level with multi-reference support and smoothing.
- **ROUGE-1/2/L** — recall-oriented unigram/bigram overlap and longest common
  subsequence; scored against the best-matching reference per image (standard
  multi-reference practice).
- **METEOR** — natively supports multiple references, and accounts for synonyms and
  stemming, making it more forgiving of valid paraphrases than BLEU/ROUGE.

## 6. Results

| Metric | Score |
|---|---|
| Best validation loss | 2.8081 (epoch 10) |
| **BLEU** | **23.22** |
| ROUGE-1 | 0.5027 |
| ROUGE-2 | 0.2529 |
| ROUGE-L | 0.4848 |
| METEOR | 0.4558 |

**Context for the BLEU score:** 23.22 is close to the BLEU-4 score reported for a
comparably-structured model (CNN + attention + LSTM) in the *Show, Attend and Tell*
paper on this same dataset (~24.3), suggesting the implementation is performing in
line with published results for this architecture family and dataset size — image
captioning BLEU scores are inherently lower than, e.g., machine translation BLEU
scores, since many different captions can correctly describe the same image.

### Qualitative examples

| Image | Predicted | Reference (sample) |
|---|---|---|
| 3324375078_9441f72898.jpg | a brown dog is playing in the snow | dogs playing in the snow |
| 2554531876_5d7f193992.jpg | a motorcycle racer is driving a turn on a race... | a motorcycle races |
| 1429546659_44cb09cbe2.jpg | two dogs are playing in a field | a white dog and a black dog in a field |
| 3241965735_8742782a70.jpg | a man is performing a trick on a bike in the snow | a parachute caught on a tree with some people |

### Error analysis / limitations

- **Dataset bias toward dogs**: the model performs noticeably better on dog-related
  scenes (a large fraction of Flickr8k images) than on rarer scene types (motorsports,
  unusual human poses), sometimes defaulting to a plausible-sounding but incorrect
  "dog in a field" style caption when the actual scene is different. This is a direct
  consequence of training on a comparatively small, imbalanced dataset.
- **Out-of-domain images**: since the model is trained exclusively on natural
  photographs, it performs poorly on stylistically different inputs (illustrations,
  paintings), often falling back to `<unk>` tokens.
- **Beam search (attempted, not used for final metrics)**: `src/inference/generate.py`
  includes a `generate_caption_beam_search` implementation as an additional decoding
  strategy. In testing, unconstrained beam search on this model produced degenerate
  repeated-word sequences (a known failure mode where a locally high-probability
  repeated token wins the length-normalized score comparison). The reported results
  above use **greedy decoding**, which was empirically more reliable at this model
  scale; a repetition-blocking constraint was identified as the fix but is noted here
  as a follow-up rather than included in the final reported numbers.

## 7. Software engineering structure

The project is organized as a Python package rather than a single notebook:

```
task6_M.I.A/
├── src/
│   ├── data/
│   │   ├── preprocessing.py   # caption cleaning, Vocabulary class
│   │   └── dataset.py          # Flickr8k loading, image/caption pairing, split
│   ├── models/
│   │   ├── encoder.py          # EfficientNet-B3 spatial feature extractor
│   │   ├── decoder.py          # LSTM + Bahdanau attention decoder
│   │   └── caption_model.py    # combines encoder + decoder
│   ├── inference/
│   │   └── generate.py         # greedy and beam-search caption generation
│   └── training/
│       ├── train.py            # training loop, checkpointing, early stopping
│       └── evaluate.py         # BLEU / ROUGE / METEOR evaluation
├── app/
│   └── gradio_app.py           # deployment interface
├── checkpoints/                 # best_model.pt, vocab.pkl
├── notebooks/                    # exploration only — not the main implementation
├── Dockerfile
├── requirements.txt
└── README.md
```

## 8. How to install and run

### Locally
```bash
pip install -r requirements.txt
python -m app.gradio_app
```
Then open `http://localhost:7860` in a browser.

### With Docker
```bash
docker build -t caption-generator .
docker run -p 7860:7860 caption-generator
```
Then open `http://localhost:7860`.

## 9. How to use the app

Upload an image through the web interface; the generated caption appears in the
"Generated caption" box. The interface runs on CPU by default if no GPU is available
in the deployment environment (inference for a single image is fast enough on CPU).

**Demo video:** https://github.com/NeSma237/M.I.A-Task-6-/blob/main/Screen%20Recording%202026-08-28%20222941.mp4

## 10. Model storage and sharing

The trained model checkpoint (`best_model.pt`) and vocabulary (`vocab.pkl`) are
publicly available on the HuggingFace Model Hub:

**https://huggingface.co/Nesmaaaa/image-caption-generator**

## 11. Training reproduction

Model and training code were developed locally (`src/`) and executed on a Kaggle
notebook (GPU-accelerated) against the Flickr8k dataset. See `src/training/train.py`
and `src/training/evaluate.py` for the exact training/evaluation entry points used.

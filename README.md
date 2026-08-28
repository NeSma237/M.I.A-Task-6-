# AI Team Training '27 — Task 6, Phase II

**MIA Robotics — AI Team**

This repository contains both parts of Task 6, Phase II: modernizing a machine
translation model (Task 1.1) and building a full end-to-end image captioning system
(Task 1.2).

## Task 1.1 — Modernizing the Machine Translation Model

An English → French Seq2Seq (BiLSTM encoder + Luong attention + LSTM decoder)
translation notebook, modernized by replacing its randomly-initialized word
embeddings with pretrained **FastText (MUSE-aligned)** vectors, and adding **BLEU**
and **ROUGE** evaluation.

- **Notebook**: [`Machine Translation.ipynb`](./Machine%20Translation.ipynb)
- **Results**: BLEU 43.94, ROUGE-1 0.69, ROUGE-2 0.51, ROUGE-L 0.68
- **Details**: see [`README.md`](./README.md) for the full write-up (approach,
  embedding choice, architecture, preprocessing, training, and results discussion).

## Task 1.2 — From Notebook to Production: Image Caption Generator

A production-oriented image captioning system trained on Flickr8k: an
EfficientNet-B3 encoder with Bahdanau attention feeding an LSTM decoder, organized
as a proper Python package (not a single notebook), with a Gradio deployment
interface, a Dockerfile, and the trained model published on HuggingFace.

- **Source code**: [`src/`](./src) — data loading & preprocessing, model (encoder /
  decoder / combined model), training, and evaluation, split into modules.
- **App**: [`app/`](./app) — Gradio interface (`gradio_app.py`).
- **Docker**: [`Dockerfile`](./Dockerfile) — builds and serves the app.
- **Trained model**: https://huggingface.co/Nesmaaaa/image-caption-generator
- **Demo video**: [Screen Recording 2026-08-28 222941.mp4](./Screen%20Recording%202026-08-28%20222941.mp4)
- **Results**: BLEU 23.22, ROUGE-1 0.50, ROUGE-2 0.25, ROUGE-L 0.48, METEOR 0.46
  (full 810-image test set)
- **Details**: see [`README (4).md`](./README%20%284%29.md) for the full write-up
  (architecture, preprocessing, training process, evaluation, error analysis, and
  how to run the app / Docker image).

## Repository structure

```
.
├── Machine Translation.ipynb   # Task 1.1 — modernized MT notebook
├── README.md                    # Task 1.1 write-up
├── README (4).md                 # Task 1.2 write-up
├── src/                           # Task 1.2 — source package
│   ├── data/                       # preprocessing.py, dataset.py
│   ├── models/                     # encoder.py, decoder.py, caption_model.py
│   ├── inference/                  # generate.py (greedy + beam search)
│   └── training/                   # train.py, evaluate.py
├── app/
│   └── gradio_app.py               # Task 1.2 deployment interface
├── checkpoints/                    # best_model.pt, vocab.pkl
├── Dockerfile                      # Task 1.2 container build
├── upload_to_hf.py                 # script used to publish the model to HuggingFace
└── requirements.txt
```

**Note:** `caption-src.zip`, `new_version.zip`, and `src.zip` are intermediate
archives created while iterating on the `src/` package on Kaggle (each upload of a
new dataset version to Kaggle required a fresh zip); they're safe to delete now that
`src/` itself is committed directly to the repo. Similarly, `task 1.2` is a stray
file from early setup and isn't part of the working project.

## How to run Task 1.2 locally

```bash
pip install -r requirements.txt
python -m app.gradio_app
```
Then open `http://localhost:7860`.

## How to run Task 1.2 with Docker

```bash
docker build -t caption-generator .
docker run -p 7860:7860 caption-generator
```
Then open `http://localhost:7860`.

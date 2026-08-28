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

- **Notebook**: [`task1.1/Machine Translation.ipynb`](./task1.1/Machine%20Translation.ipynb)
- **Results**: BLEU 43.94, ROUGE-1 0.69, ROUGE-2 0.51, ROUGE-L 0.68
- **Details**: see [`task1.1/README.md`](./task1.1/README.md) for the full write-up
  (approach, embedding choice, architecture, preprocessing, training, and results
  discussion).

## Task 1.2 — From Notebook to Production: Image Caption Generator

A production-oriented image captioning system trained on Flickr8k: an
EfficientNet-B3 encoder with Bahdanau attention feeding an LSTM decoder, organized
as a proper Python package (not a single notebook), with a Gradio deployment
interface, a Dockerfile, and the trained model published on HuggingFace.

- **Source code**: [`task1.2/src/`](./task1.2/src) — data loading & preprocessing,
  model (encoder / decoder / combined model), training, and evaluation, split into
  modules.
- **App**: [`task1.2/app/`](./task1.2/app) — Gradio interface (`gradio_app.py`).
- **Docker**: [`task1.2/Dockerfile`](./task1.2/Dockerfile) — builds and serves the
  app.
- **Trained model**: https://huggingface.co/Nesmaaaa/image-caption-generator
- **Results**: BLEU 23.22, ROUGE-1 0.50, ROUGE-2 0.25, ROUGE-L 0.48, METEOR 0.46
  (full 810-image test set)
- **Details**: see [`task1.2/README (4).md`](./task1.2/README%20%284%29.md) for the
  full write-up (architecture, preprocessing, training process, evaluation, error
  analysis, and how to run the app / Docker image).

## Repository structure

```
.
├── task1.1/
│   ├── Machine Translation.ipynb   # Task 1.1 — modernized MT notebook
│   └── README.md                    # Task 1.1 write-up
├── task1.2/
│   ├── src/                          # Task 1.2 — source package
│   │   ├── data/                       # preprocessing.py, dataset.py
│   │   ├── models/                     # encoder.py, decoder.py, caption_model.py
│   │   ├── inference/                  # generate.py (greedy + beam search)
│   │   └── training/                   # train.py, evaluate.py
│   ├── app/
│   │   └── gradio_app.py               # Task 1.2 deployment interface
│   ├── checkpoints/                    # vocab.pkl, config.json (model weights on HF)
│   ├── notebooks/                      # exploratory / Kaggle notebooks
│   ├── Dockerfile                      # Task 1.2 container build
│   ├── upload_to_hf.py                 # script used to publish the model to HuggingFace
│   ├── requirements.txt
│   └── README (4).md                   # Task 1.2 write-up
└── README.md                         # this file
```

## How to run Task 1.2 locally

```bash
cd task1.2
pip install -r requirements.txt
python -m app.gradio_app
```
Then open `http://localhost:7860`.

## How to run Task 1.2 with Docker

```bash
cd task1.2
docker build -t caption-generator .
docker run -p 7860:7860 caption-generator
```
Then open `http://localhost:7860`.

# Task 1.1 — Modernizing the Machine Translation Model

**AI Team Training '27 — Task 6, Phase II**

This project modernizes the workshop's English → French Seq2Seq translation model by
replacing its randomly-initialized word embeddings with **pretrained FastText
(MUSE-aligned) vectors**, and adding **BLEU** and **ROUGE** evaluation on top of the
original token-accuracy metric. The rest of the architecture (BiLSTM encoder, Luong
attention, LSTM decoder, training loop) is unchanged from the workshop notebook.

---

## 1. Overall approach

The workshop notebook builds an English → French neural machine translation model
from scratch in PyTorch, using randomly-initialized embeddings that are learned
purely from the ~136k Tatoeba/Anki sentence pairs. The goal of this task was to:

1. Replace the frequency-based / random embeddings with a modern **pretrained
   word-embedding method** (FastText).
2. Keep the existing BiLSTM + attention + LSTM decoder architecture unchanged.
3. Evaluate translation quality with **BLEU** and **ROUGE**, not just token accuracy.

## 2. Word-embedding method: FastText (MUSE-aligned)

**Why FastText over Word2Vec / GloVe:** FastText represents words as bags of
character n-grams, so it can produce a reasonable vector for out-of-vocabulary or
rare words instead of falling back to `<unk>`. This matters more for French than
English, since French has heavy verb conjugation and gender agreement, producing
many word forms a fixed vocabulary can miss.

**Additional technique — cross-lingual alignment:** instead of using two unrelated
FastText models for English and French, we use the **MUSE-aligned** release, where
English and French vectors live in the **same 300-dimensional semantic space**. A
word like `cat` (EN) and `chat` (FR) start out close together before any training
happens, giving the encoder and decoder a shared starting point instead of two
independent random spaces.

**Fine-tuning, not freezing:** the embeddings are loaded via
`nn.Embedding.from_pretrained(..., freeze=False)`, so they continue to update during
training. This was a deliberate choice: FastText was trained on Wikipedia text, while
our corpus is short, casual, spoken-style sentences (Tatoeba) — letting the model
adjust the vectors closes that domain gap.

**Vocabulary coverage achieved:**

| Language | Vocab size | Words found in FastText | Coverage |
|---|---|---|---|
| English (source) | 10,000 | 9,857 | **98.6%** |
| French (target) | 10,000 | 9,277 | **92.8%** |

French coverage is lower than English, consistent with its richer morphology and the
gap between the Wikipedia-trained vectors and casual conversational French. Words not
found in FastText keep a small random vector and are learned from data, exactly as in
the original notebook.

## 3. Architecture: BiLSTM encoder + Luong attention + LSTM decoder

The architecture itself was **not modified** — only the embedding layer's
initialization changed. For completeness:

- **Encoder** — a single-layer **bidirectional LSTM** reads the English sentence in
  both directions and concatenates the forward/backward hidden states
  (`hidden_dim=256` per direction → 512 combined), so each position's representation
  reflects context from both sides of the sentence.
- **Attention (Luong / dot-product)** — at each decoding step, the decoder's hidden
  state is compared against every encoder output via a dot product, turned into a
  probability distribution with softmax (masked so padding positions are never
  attended to), and used to compute a weighted "context" vector — i.e. which English
  words are relevant to the French word being generated right now.
- **Decoder** — a single-layer LSTM generates French tokens one at a time. At each
  step, its output is concatenated with the attention context vector and projected
  to the French vocabulary via a linear layer.
- Embedding dimension increased from 256 → **300** to match the pretrained FastText
  vectors; all other hyperparameters (`hidden_dim=256`, `decoder_hidden=512`,
  `batch_size=64`, `dropout=0.2`) are unchanged from the workshop configuration.

## 4. Preprocessing and training process

- **Data**: 135,842 English–French sentence pairs (Tatoeba/Anki), downloaded
  automatically from `download.pytorch.org` if not present locally.
- **Cleaning**: lowercasing, contraction expansion (`don't` → `do not`,
  `c'est` → `ce est`), and stripping non-alphabetic characters (keeping French
  accented letters).
- **Split**: 81% train / 9% validation / 10% test, with a fixed random seed for
  reproducibility.
- **Vocabulary**: word-level, capped at 10,000 tokens per language (both languages
  reached the cap), with `<pad>`, `<unk>`, `<start>`, `<end>` special tokens.
- **Training**: teacher forcing, Adam optimizer (`lr=1e-3`), `ReduceLROnPlateau`
  scheduler, gradient clipping (`max_norm=1.0`), and early stopping
  (`patience=3` on validation loss).
- **Result**: training stopped early at **epoch 11** (best weights from **epoch 8**,
  validation loss 1.1316), after 22.8 minutes on a Tesla T4 GPU.

| Epoch | Train loss | Val loss | Train acc | Val acc |
|---|---|---|---|---|
| 1 | 2.9248 | 1.7204 | 0.491 | 0.647 |
| 4 | 0.8002 | 1.1594 | 0.789 | 0.738 |
| **8 (best)** | 0.3901 | **1.1316** | 0.879 | 0.757 |
| 11 (stopped) | 0.2590 | 1.1688 | 0.914 | 0.758 |

Train and validation accuracy diverge clearly after epoch ~4–5 (91.4% vs. 75.8% by
epoch 11), indicating mild **overfitting** — training accuracy keeps climbing while
validation accuracy plateaus. This is expected given the model size relative to a
10k-word vocabulary, and is why early stopping restored the epoch-8 weights rather
than the final epoch.

## 5. Evaluation metrics

Token accuracy (used during training) only measures how often the model predicts the
correct *next* token under teacher forcing — it says nothing about whether a freely
generated translation is actually good. We therefore also evaluate fully generated
translations (greedy decoding) against the reference sentences using:

- **BLEU** (Bilingual Evaluation Understudy) — n-gram precision (1- to 4-gram) with a
  brevity penalty; the standard MT metric. Computed with NLTK's `corpus_bleu` and
  smoothing (method4), since our sentences are short.
- **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) — ROUGE-1/2 measure
  unigram/bigram overlap, ROUGE-L measures the longest common subsequence (rewarding
  correct word order without requiring an exact contiguous match). A recall-oriented
  complement to BLEU's precision focus.

## 6. Results

Evaluated on a random sample of **1,000** test sentences (greedy decoding one
sentence at a time is slow over the full ~13.6k test set):

| Metric | Score |
|---|---|
| Test loss (teacher forcing) | 1.1196 |
| Test token accuracy | 0.759 |
| **BLEU** | **43.94** |
| ROUGE-1 | 0.6912 |
| ROUGE-2 | 0.5096 |
| ROUGE-L | 0.6819 |

A BLEU score in the 40–50 range is a strong result for a BiLSTM+attention model of
this size on a casual-speech dataset (scores in this range are generally considered
good/fluent, as opposed to <20 which indicates barely-comprehensible output). The
relatively high ROUGE-1/ROUGE-L scores confirm that the model's word choice and word
*order* closely track the reference translations.

### Qualitative examples

| English | Reference (FR) | Predicted (FR) |
|---|---|---|
| he made up a story about the dog | il inventa une histoire au sujet du chien | il a inventé une histoire au sujet du chien |
| can you determine what happened | peux tu déterminer ce qui s est passé | peux tu déterminer ce qui s est passé |
| tom is always watching tv | tom regarde toujours la télévision | tom regarde toujours la télévision |
| there is nothing like cold beer on a hot day | il n y a rien de tel qu une bière fraîche un jour de canicule | il n y a rien de frais d eau sur un jour chaud |
| my children do not listen to me | mes enfants ne m écoutent pas | mes enfants m adorent |

**Idiomatic translation example:** for `"it is pitch black outside"`, the model
generated `"il fait noir comme dans un four dehors"` — using the real French idiom
*"noir comme dans un four"* ("black as inside an oven") rather than a literal
word-for-word translation. The attention heatmap confirms this: the idiom's words all
attend strongly back to `black`, showing the model generates the expression as a
unit rather than token-by-token.

### Error analysis

- **Word-choice errors under semantic similarity**: *"cold beer"* → *"de frais
  d'eau"* ("cold/fresh water") instead of *"bière fraîche"*. The model correctly
  captured "something cold and refreshing on a hot day" but picked the wrong
  specific noun — a content error, not a grammar error.
- **Meaning inversion on rarer patterns**: *"do not listen to me"* → *"m'adorent"*
  ("adore me"), the opposite of the intended meaning — likely a rarer phrasing
  confused with a more frequent nearby pattern in training.
- **Early cutoff on loanwords**: *"Where is the parking?"* → *"où est le"* (sentence
  cut short). `parking` is an English loanword used in French but likely rare in the
  training data, so the model appears to "give up" and predict `<end>` early rather
  than risk an uncertain word — a limitation of greedy decoding.
- **Minor gender-agreement slips**: e.g. *"heureuse"* (fem.) in the reference vs.
  *"heureux"* (masc.) generated — a common, low-impact French grammar error.

## 7. Summary

Swapping random embeddings for pretrained, cross-lingually aligned FastText vectors
gave the model a meaningful head start (>92% vocabulary coverage in both languages)
without any architecture changes. The resulting model reaches a **BLEU score of
43.94** and produces fluent, often idiomatically correct French translations, with
most errors being specific word-choice mistakes rather than structural or grammatical
failures — visible both in the qualitative examples and in the gap between the high
ROUGE-1/ROUGE-L (word/order overlap) and lower ROUGE-2 (exact bigram overlap) scores.

"""Evaluation on unseen test images: BLEU, ROUGE, METEOR — each image is scored
against *all* of its reference captions, not just one."""

import nltk
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from src.inference.generate import generate_caption_beam_search

def _ensure_nltk_data():
    for pkg in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def evaluate_model(model, test_df: pd.DataFrame, images_dir: str, vocab,
                    device, n_samples: int = None, seed: int = 42):
    """
    Args:
        test_df: DataFrame with columns [image, caption] — multiple rows per
            image (one per reference caption), as produced by dataset.py
        n_samples: if set, evaluate on this many unique images instead of all
            of them (greedy decoding is slow one image at a time)

    Returns:
        dict of metric name -> score, plus a DataFrame of per-image examples
        for qualitative inspection.
    """
    _ensure_nltk_data()

    # Group all reference captions per image — Flickr8k has 5 per image, and
    # BLEU/METEOR should be computed against the full reference set, not just
    # whichever single caption happened to be in a given row.
    grouped = test_df.groupby("image")["caption"].apply(list).reset_index()
    if n_samples is not None:
        grouped = grouped.sample(min(n_samples, len(grouped)), random_state=seed)

    references_list = []   # list[list[list[str]]]  (per image: list of ref token-lists)
    hypotheses = []         # list[list[str]]
    rows = []                # for the qualitative examples table

    for _, row in grouped.iterrows():
        image_path = f"{images_dir}/{row['image']}"
        pred = generate_caption_beam_search(model, image_path, vocab, device, beam_width=5)
        refs_tokens = [ref.split() for ref in row["caption"]]
        references_list.append(refs_tokens)
        hypotheses.append(pred.split())

        rows.append({
            "image": row["image"],
            "predicted": pred,
            "references": row["caption"],
        })

    # --- BLEU (corpus-level, multi-reference) ---
    smoothie = SmoothingFunction().method4
    bleu = corpus_bleu(references_list, hypotheses, smoothing_function=smoothie)

    # --- ROUGE (averaged per-image, against the best-matching reference) ---
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    rouge1, rouge2, rougeL = [], [], []
    for refs_tokens, hyp_tokens in zip(references_list, hypotheses):
        hyp_text = " ".join(hyp_tokens)
        # score against every reference, keep the best (standard multi-ref ROUGE practice)
        best = max(
            (scorer.score(" ".join(ref), hyp_text) for ref in refs_tokens),
            key=lambda s: s["rougeL"].fmeasure,
        )
        rouge1.append(best["rouge1"].fmeasure)
        rouge2.append(best["rouge2"].fmeasure)
        rougeL.append(best["rougeL"].fmeasure)

    # --- METEOR (natively supports multiple references per hypothesis) ---
    meteor_scores = [
        meteor_score(refs_tokens, hyp_tokens)
        for refs_tokens, hyp_tokens in zip(references_list, hypotheses)
    ]

    metrics = {
        "n_images": len(hypotheses),
        "BLEU": bleu * 100,
        "ROUGE-1": sum(rouge1) / len(rouge1),
        "ROUGE-2": sum(rouge2) / len(rouge2),
        "ROUGE-L": sum(rougeL) / len(rougeL),
        "METEOR": sum(meteor_scores) / len(meteor_scores),
    }

    examples_df = pd.DataFrame(rows)
    return metrics, examples_df


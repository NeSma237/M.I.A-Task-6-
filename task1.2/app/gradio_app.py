"""Gradio interface: upload an image, get a generated caption. Also serves as
the entry point for the Docker container."""

import pickle
import tempfile
from pathlib import Path

import gradio as gr
import torch

from src.data.preprocessing import Vocabulary
from src.inference.generate import generate_caption
from src.models.caption_model import CaptionModel

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best_model.pt"
VOCAB_PATH = BASE_DIR / "checkpoints" / "vocab.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_vocab():
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    model = CaptionModel(
        vocab_size=len(vocab),
        embed_dim=256,
        decoder_dim=512,
        attention_dim=512,
        fine_tune_from_block=6,
        dropout=0.5,
        pad_idx=vocab.pad_idx,
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f})")
    return model, vocab


model, vocab = load_model_and_vocab()


def caption_image(image):
    """Gradio callback: receives a PIL image directly (type='pil'), saves it
    to a temp path since generate_caption expects a file path, then runs
    inference."""
    if image is None:
        return "Please upload an image first."

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_path = tmp.name
    image.convert("RGB").save(temp_path)

    caption = generate_caption(model, temp_path, vocab, device)
    return caption


demo = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Textbox(label="Generated caption"),
    title="Image Caption Generator",
    description=(
        "Upload a photo and get an automatically generated English caption. "
        "Trained on Flickr8k using an EfficientNet-B3 encoder, Bahdanau "
        "attention, and an LSTM decoder."
    ),
    examples=None,  # optionally: add a few sample image paths here
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
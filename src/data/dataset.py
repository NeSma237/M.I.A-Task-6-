"""Flickr8k loading, image/caption pairing, and train/val/test split (by image,
to avoid data leakage between splits)."""

import os
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.preprocessing import clean_caption, Vocabulary

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 300, train: bool = True) -> transforms.Compose:
    """Light augmentation for training, deterministic resize for val/test/inference.
    image_size=300 matches EfficientNet-B3's expected input resolution."""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_captions(captions_path: str) -> pd.DataFrame:
    """Load Flickr8k captions into a DataFrame with columns [image, caption].

    Supports both common release formats:
      - captions.txt  : 'image,caption' (comma-separated, header row)
      - Flickr8k.token.txt : 'image.jpg#0\\tcaption' (tab-separated, no header)
    """
    with open(captions_path, encoding="utf-8") as f:
        first_line = f.readline()

    if "," in first_line and "#" not in first_line:
        df = pd.read_csv(captions_path)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={df.columns[0]: "image", df.columns[1]: "caption"})
    else:
        rows = []
        with open(captions_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_tag, caption = line.split("\t")
                image = img_tag.split("#")[0]
                rows.append((image, caption))
        df = pd.DataFrame(rows, columns=["image", "caption"])

    df["caption"] = df["caption"].map(clean_caption)
    df = df[df["caption"].str.len() > 0].reset_index(drop=True)
    return df


def split_by_image(df: pd.DataFrame, seed: int, test_size: float = 0.10,
                    val_size: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split train/val/test by unique image filename, so all 5 captions of a given
    image always land in the same split (avoids leakage across splits)."""
    unique_images = df["image"].unique()

    train_imgs, test_imgs = train_test_split(unique_images, test_size=test_size, random_state=seed)
    train_imgs, val_imgs = train_test_split(train_imgs, test_size=val_size, random_state=seed)

    train_df = df[df["image"].isin(train_imgs)].reset_index(drop=True)
    val_df = df[df["image"].isin(val_imgs)].reset_index(drop=True)
    test_df = df[df["image"].isin(test_imgs)].reset_index(drop=True)
    return train_df, val_df, test_df


class Flickr8kDataset(Dataset):
    """One (image, single caption) pair per item — an image with 5 captions
    appears 5 times, each with a different caption, so the model sees every
    reference during training."""

    def __init__(self, df: pd.DataFrame, images_dir: str, vocab: Vocabulary,
                 max_len: int, transform: transforms.Compose):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.images_dir / row["image"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        caption_ids, length = self.vocab.encode(row["caption"], self.max_len)
        return {
            "image": image,
            "caption": caption_ids,
            "length": length,
            "image_file": row["image"],
            "raw_caption": row["caption"],
        }
"""Training loop: loss, optimizer, LR scheduling, early stopping, checkpointing."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.models.caption_model import CaptionModel


def collate_fn(batch):
    """Stacks a list of dataset items into batch tensors."""
    images = torch.stack([item["image"] for item in batch])
    captions = torch.tensor([item["caption"] for item in batch], dtype=torch.long)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    return images, captions, lengths


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 2) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def run_epoch(model: CaptionModel, loader: DataLoader, criterion, optimizer,
              device, pad_idx: int, train: bool, grad_clip: float = 5.0,
              alpha_c: float = 1.0):
    """One pass over the data. `alpha_c` weights the doubly-stochastic attention
    regularization (see note below)."""
    model.train() if train else model.eval()

    total_loss, n_batches = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, captions, lengths in loader:
            images = images.to(device)
            captions = captions.to(device)
            lengths = lengths.to(device)

            logits, alphas = model(images, captions, lengths)
            targets = captions[:, 1:]  # predict everything after <start>

            loss = criterion(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

            # Doubly stochastic attention regularization (Show, Attend and Tell):
            # encourages the sum of attention weights for each image region,
            # across all decoding steps, to be close to 1 — i.e. every part of
            # the image should get attended to at some point, not just one
            # region repeatedly. Improves caption diversity and reduces the
            # model fixating on a single salient object.
            reg_loss = alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
            loss = loss + reg_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def train_model(model: CaptionModel, train_loader, val_loader, device,
                 pad_idx: int, epochs: int = 20, lr: float = 3e-4,
                 patience: int = 4, checkpoint_dir: str = "checkpoints",
                 checkpoint_name: str = "best_model.pt"):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    stale = 0
    t0 = time.time()

    print(f"{'epoch':>6}  {'train_loss':>11}  {'val_loss':>9}  {'time':>6}")
    print("-" * 42)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss = run_epoch(model, train_loader, criterion, optimizer,
                                device, pad_idx, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer,
                              device, pad_idx, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        elapsed = time.time() - epoch_start
        print(f"{epoch:6d}  {train_loss:11.4f}  {val_loss:9.4f}  {elapsed:5.0f}s")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            stale = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, checkpoint_path / checkpoint_name)
            print(f"  ↳ saved new best checkpoint (val_loss={val_loss:.4f})")
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch} (best val loss = {best_val:.4f})")
                break

    print(f"\nFinished in {(time.time() - t0) / 60:.1f} min. Best val loss: {best_val:.4f}")

    # Reload best weights before returning, so the caller always has the best model
    best_ckpt = torch.load(checkpoint_path / checkpoint_name, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    return model, history
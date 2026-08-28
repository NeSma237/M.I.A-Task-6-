"""Greedy and beam-search caption generation for a single image at inference
time (no ground truth caption available — unlike training's teacher forcing)."""

import torch
from PIL import Image

from src.data.dataset import build_transforms
from src.data.preprocessing import Vocabulary
from src.models.caption_model import CaptionModel


@torch.no_grad()
def generate_caption(model: CaptionModel, image_path: str, vocab: Vocabulary,
                      device, max_len: int = 30, return_attention: bool = False):
    """Greedy decoding — picks the single best word at every step. Fast, but can
    get stuck in locally-good-but-globally-wrong choices (e.g. repeating a word)."""
    model.eval()

    transform = build_transforms(train=False)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    encoder_out = model.encoder(image_tensor)
    num_pixels = encoder_out.size(1)

    h, c = model.decoder.init_hidden_state(encoder_out)
    token = torch.tensor([vocab.sos_idx], device=device)

    output_ids = []
    attn_steps = []

    for _ in range(max_len):
        embedding = model.decoder.embedding(token)
        context, alpha = model.decoder.attention(encoder_out, h)
        gate = model.decoder.sigmoid(model.decoder.f_beta(h))
        context = gate * context
        lstm_input = torch.cat([embedding, context], dim=1)
        h, c = model.decoder.lstm_cell(lstm_input, (h, c))
        logits = model.decoder.fc(h)
        next_id = int(logits.argmax(dim=1))
        attn_steps.append(alpha.squeeze(0).cpu())

        if next_id == vocab.eos_idx:
            break
        output_ids.append(next_id)
        token = torch.tensor([next_id], device=device)

    caption = vocab.decode(output_ids)

    if return_attention:
        attn_mat = torch.stack(attn_steps, dim=0) if attn_steps else torch.zeros(1, num_pixels)
        grid_size = int(num_pixels ** 0.5)
        return caption, attn_mat, grid_size

    return caption


@torch.no_grad()
def generate_caption_beam_search(model: CaptionModel, image_path: str,
                                  vocab: Vocabulary, device, beam_width: int = 5,
                                  max_len: int = 30):
    """Beam search decoding: instead of committing to the single best word at
    each step, keeps the top `beam_width` partial sequences alive at every
    step, and only commits at the end to whichever complete sequence has the
    highest total (length-normalized) log-probability."""
    model.eval()

    transform = build_transforms(train=False)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    encoder_out = model.encoder(image_tensor)

    h0, c0 = model.decoder.init_hidden_state(encoder_out)

    beams = [([vocab.sos_idx], 0.0, h0, c0)]
    completed = []

    for _ in range(max_len):
        candidates = []
        for token_ids, log_prob, h, c in beams:
            last_token = token_ids[-1]
            if last_token == vocab.eos_idx:
                completed.append((token_ids, log_prob))
                continue

            token = torch.tensor([last_token], device=device)
            embedding = model.decoder.embedding(token)
            context, _ = model.decoder.attention(encoder_out, h)
            gate = model.decoder.sigmoid(model.decoder.f_beta(h))
            context = gate * context
            lstm_input = torch.cat([embedding, context], dim=1)
            new_h, new_c = model.decoder.lstm_cell(lstm_input, (h, c))
            logits = model.decoder.fc(new_h)
            log_probs = torch.log_softmax(logits, dim=1).squeeze(0)

            top_log_probs, top_ids = log_probs.topk(beam_width)
            for lp, idx in zip(top_log_probs.tolist(), top_ids.tolist()):
                candidates.append((token_ids + [idx], log_prob + lp, new_h, new_c))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
        beams = candidates[:beam_width]

        if len(completed) >= beam_width:
            break

    completed.extend([(t, lp) for t, lp, _, _ in beams])
    best_ids, _ = max(completed, key=lambda x: x[1] / len(x[0]))

    output_ids = [i for i in best_ids if i not in (vocab.sos_idx, vocab.eos_idx)]
    return vocab.decode(output_ids)
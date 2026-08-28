"""Text preprocessing utilities: cleaning and vocabulary building for captions."""

import re
from collections import Counter


def clean_caption(text: str) -> str:
    """Lowercase, strip punctuation/digits, collapse whitespace."""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class Vocabulary:
    """Word-level vocabulary with reserved special tokens and a min-frequency cutoff
    to handle rare words (task requirement: 'handling unknown and rare words').
    """

    PAD = "<pad>"
    UNK = "<unk>"
    SOS = "<start>"
    EOS = "<end>"

    def __init__(self, min_freq: int = 5):
        self.min_freq = min_freq
        self.word2idx = {self.PAD: 0, self.UNK: 1, self.SOS: 2, self.EOS: 3}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def build(self, captions: list[str]) -> "Vocabulary":
        counts = Counter()
        for caption in captions:
            counts.update(caption.split())
        # Only keep words seen at least `min_freq` times — rarer words become <unk>
        for word, freq in counts.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        return self

    def encode(self, caption: str, max_len: int) -> tuple[list[int], int]:
        tokens = [self.SOS] + caption.split() + [self.EOS]
        ids = [self.word2idx.get(tok, self.word2idx[self.UNK]) for tok in tokens]
        ids = ids[:max_len]
        length = len(ids)
        ids = ids + [self.word2idx[self.PAD]] * (max_len - length)
        return ids, length

    def decode(self, ids: list[int]) -> str:
        words = []
        for idx in ids:
            word = self.idx2word.get(int(idx), self.UNK)
            if word == self.EOS:
                break
            if word in (self.PAD, self.SOS):
                continue
            words.append(word)
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word2idx)

    @property
    def pad_idx(self) -> int:
        return self.word2idx[self.PAD]

    @property
    def sos_idx(self) -> int:
        return self.word2idx[self.SOS]

    @property
    def eos_idx(self) -> int:
        return self.word2idx[self.EOS]
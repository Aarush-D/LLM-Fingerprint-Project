import re
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def tokenize(text):
    text = str(text).lower()
    return re.findall(r"\b\w+\b|[^\w\s]", text)

def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def encode_text(text, vocab, max_len=300):
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens][:max_len]
    if not ids:
        ids = [vocab[UNK_TOKEN]]
    return torch.tensor(ids, dtype=torch.long)

def collate_batch(batch, pad_idx=0):
    xs, ys = zip(*batch)
    xs = pad_sequence(xs, batch_first=True, padding_value=pad_idx)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, ys

import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from preprocess import build_vocab, encode_text, collate_batch
from models import TextCNN, LSTMClassifier

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=300):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = encode_text(self.texts[idx], self.vocab, self.max_len)
        y = self.labels[idx]
        return x, y

def make_input(df, mode):
    if mode == "input":
        return df["LLM_Input"].fillna("").astype(str)
    if mode == "output":
        return df["LLM_output"].fillna("").astype(str)
    if mode == "both":
        return (
            "USER PROMPT: " + df["LLM_Input"].fillna("").astype(str)
            + " MODEL RESPONSE: " + df["LLM_output"].fillna("").astype(str)
        )
    raise ValueError("mode must be input, output, or both")

def evaluate(model, loader, device):
    model.eval()
    preds, gold = [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            preds.extend(logits.argmax(1).cpu().tolist())
            gold.extend(y.cpu().tolist())
    return total_loss / len(loader.dataset), accuracy_score(gold, preds), gold, preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/llm_dataset.csv")
    parser.add_argument("--model", choices=["cnn", "lstm"], required=True)
    parser.add_argument("--mode", choices=["input", "output", "both"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=300)
    args = parser.parse_args()

    set_seed()
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    df = pd.read_csv(args.data)
    required = {"LLM_name", "LLM_Input", "LLM_output"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    df = df.dropna(subset=["LLM_name"]).copy()
    df["text"] = make_input(df, args.mode)

    labels = sorted(df["LLM_name"].unique())
    label_to_id = {name: i for i, name in enumerate(labels)}
    df["label"] = df["LLM_name"].map(label_to_id)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED
    )

    vocab = build_vocab(train_df["text"])

    train_ds = TextDataset(train_df["text"], train_df["label"], vocab, args.max_len)
    val_ds = TextDataset(val_df["text"], val_df["label"], vocab, args.max_len)
    test_ds = TextDataset(test_df["text"], test_df["label"], vocab, args.max_len)

    collate = lambda batch: collate_batch(batch, pad_idx=0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    if args.model == "cnn":
        model = TextCNN(len(vocab), len(labels))
    else:
        model = LSTMClassifier(len(vocab), len(labels))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)

        train_loss = running / len(train_loader.dataset)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.4f}")

    test_loss, test_acc, gold, preds = evaluate(model, test_loader, device)
    print(f"\nTEST ACCURACY: {test_acc:.4f}\n")
    print(classification_report(gold, preds, target_names=labels, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(gold, preds))

    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)

    tag = f"{args.model}_{args.mode}"
    torch.save(model.state_dict(), results_dir / f"{tag}.pt")

    pd.DataFrame({
        "epoch": list(range(1, args.epochs + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_accuracy": val_accs,
    }).to_csv(results_dir / f"{tag}_history.csv", index=False)

    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train loss")
    plt.plot(range(1, args.epochs + 1), val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{args.model.upper()} - {args.mode}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f"{tag}_loss.png", dpi=200)
    plt.close()

    with open(results_dir / f"{tag}_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Labels: {labels}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write("\nClassification report:\n")
        f.write(classification_report(gold, preds, target_names=labels, digits=4))
        f.write("\nConfusion matrix:\n")
        f.write(str(confusion_matrix(gold, preds)))

if __name__ == "__main__":
    main()

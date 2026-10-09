import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=128, num_filters=128, kernel_size=3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)          # [B, L, E]
        x = x.transpose(1, 2)          # [B, E, L]
        x = self.relu(self.conv(x))     # [B, F, L']
        x = self.pool(x).squeeze(-1)    # [B, F]
        return self.fc(x)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=128, hidden_dim=128, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

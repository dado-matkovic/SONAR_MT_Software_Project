import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeechEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, proj_dim=256, n_layers=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(input_dim, proj_dim)

    def forward(self, feats, feats_len):
        enc = self.encoder(feats)
        mask = torch.arange(enc.size(1), device=feats.device).unsqueeze(0) < feats_len.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        enc = enc * mask
        pooled = enc.sum(dim=1) / mask.sum(dim=1)
        return F.normalize(self.proj(pooled), p=2, dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, proj_dim=256, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(hidden_dim, proj_dim)

    def forward(self, text, text_len):
        emb = self.embed(text)
        enc = self.encoder(emb)
        mask = torch.arange(enc.size(1), device=enc.device).unsqueeze(0) < text_len.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        enc = enc * mask
        pooled = enc.sum(dim=1) / mask.sum(dim=1)
        return F.normalize(self.proj(pooled), p=2, dim=-1)


def contrastive_loss(speech_emb, text_emb, temperature=0.07):
    speech_emb = F.normalize(speech_emb, p=2, dim=-1)
    text_emb = F.normalize(text_emb, p=2, dim=-1)
    logits = torch.matmul(speech_emb, text_emb.T) / temperature
    labels = torch.arange(len(logits)).to(logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2

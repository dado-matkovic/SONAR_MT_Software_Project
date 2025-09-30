import os
import yaml
import torch
from torch.utils.data import DataLoader
from torch import optim
import sentencepiece as spm

from dataset import SpeechTextDataset, collate_fn
from models import SpeechEncoder, TextEncoder, contrastive_loss
from utils import set_seed

# -------------------
# Load config
# -------------------
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Fix numeric parsing issues
cfg["optim"]["lr"] = float(cfg["optim"]["lr"])
cfg["optim"]["weight_decay"] = float(cfg["optim"]["weight_decay"])
cfg["optim"]["betas"] = tuple(float(x) for x in cfg["optim"]["betas"])

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Dataset + DataLoader
# -------------------
ds = SpeechTextDataset(cfg["manifest"], cfg["spm_model"], sample_rate=cfg["audio"]["sample_rate"])

if cfg.get("debug", False):
    ds.data = ds.data.head(500)

loader = DataLoader(
    ds,
    batch_size=int(cfg["train"]["batch_size"]),
    shuffle=True,
    num_workers=int(cfg["train"]["num_workers"]),
    collate_fn=collate_fn,
)

# -------------------
# Models
# -------------------
speech_encoder = SpeechEncoder(
    hidden_dim=int(cfg["model"]["hidden_dim"]),
    proj_dim=int(cfg["model"]["proj_dim"]),
    n_layers=int(cfg["model"]["n_layers"]),
).to(device)

# vocab size from sentencepiece
sp = spm.SentencePieceProcessor(model_file=cfg["spm_model"])
vocab_size = sp.get_piece_size()

text_encoder = TextEncoder(
    vocab_size=vocab_size,
    hidden_dim=int(cfg["model"]["hidden_dim"]),
    proj_dim=int(cfg["model"]["proj_dim"]),
    n_layers=2,
).to(device)

params = list(speech_encoder.parameters()) + list(text_encoder.parameters())
optimizer = optim.AdamW(
    params,
    lr=cfg["optim"]["lr"],
    weight_decay=cfg["optim"]["weight_decay"],
    betas=cfg["optim"]["betas"]
)

# -------------------
# Training Loop
# -------------------
epochs = int(cfg["train"]["epochs"])
out_dir = cfg["paths"]["out_dir"]
os.makedirs(out_dir, exist_ok=True)

for epoch in range(epochs):
    speech_encoder.train()
    text_encoder.train()
    total_loss = 0

    for batch in loader:
        feats = batch["feats"].to(device)        # <-- changed
        feats_len = batch["feats_len"].to(device)  # <-- changed
        text = batch["text_ids"].to(device)
        text_len = batch["text_len"].to(device)

        optimizer.zero_grad()

        speech_emb = speech_encoder(feats, feats_len)  # <-- changed
        text_emb = text_encoder(text, text_len)

        loss = contrastive_loss(speech_emb, text_emb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # -------------------
    # Save lightweight checkpoint (only last 2 kept)
    # -------------------
    ckpt_path = os.path.join(out_dir, f"model_epoch{epoch+1}.pt")
    torch.save(
        {
            "speech_encoder": speech_encoder.state_dict(),
            "text_encoder": text_encoder.state_dict(),
            "epoch": epoch + 1,
        },
        ckpt_path,
    )

    # Keep only last 2 checkpoints
    checkpoints = sorted([f for f in os.listdir(out_dir) if f.endswith(".pt")])
    if len(checkpoints) > 2:
        os.remove(os.path.join(out_dir, checkpoints[0]))

    print(f"✅ Saved checkpoint {ckpt_path} (only last 2 kept)")


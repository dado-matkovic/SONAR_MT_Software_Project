import os
import random
import numpy as np
import torch
import sentencepiece as spm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_spm(spm_path):
    return spm.SentencePieceProcessor(model_file=spm_path)

def save_checkpoint(state, out_dir, name="model.pt"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    torch.save(state, path)
    print(f" Checkpoint saved here: {path}")

def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    return ckpt.get("epoch", 0)

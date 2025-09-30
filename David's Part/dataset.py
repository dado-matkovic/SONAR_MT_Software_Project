import os
import torch
import torchaudio
import pandas as pd
import sentencepiece as spm

class SpeechTextDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, spm_model, sample_rate=16000, n_mels=80, hop_length=160):
        self.data = pd.read_csv(manifest_path, sep="\t")
        self.sample_rate = sample_rate
        self.spm = spm.SentencePieceProcessor(model_file=spm_model)

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        self.base_dir = os.path.dirname(os.path.abspath(manifest_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = row["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(self.base_dir, audio_path)

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"No file here: {audio_path}")



        wav, sr = torchaudio.load(audio_path)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        features = self.melspec(wav).squeeze(0).transpose(0, 1)
        features = torch.log(features + 1e-6)

        ids = self.spm.encode(str(row["text"]), out_type=int)
        ids = torch.tensor(ids, dtype=torch.long)

        return {
            "features": features,                      
            "feats_len": torch.tensor(len(features)),
            "text_ids": ids,                     
            "text_len": torch.tensor(len(ids)),
        }


def collate_fn(batch):
    # Feature padding
    features = [b["features"] for b in batch]
    feats_lens = torch.tensor([len(f) for f in features], dtype=torch.long)
    max_feats_len = max(feats_lens)
    n_mels = features[0].shape[1]

    feats_padded = torch.zeros(len(batch), max_feats_len, n_mels)
    for i, f in enumerate(features):
        feats_padded[i, : len(f)] = f

    # Text padding
    texts = [b["text_ids"] for b in batch]
    text_lens = torch.tensor([len(t) for t in texts], dtype=torch.long)
    max_text_len = max(text_lens)
    pad_id = 0  # SentencePiece usually reserves id=0 as <pad>
    text_padded = torch.full((len(batch), max_text_len), pad_id, dtype=torch.long)
    for i, t in enumerate(texts):
        text_padded[i, : len(t)] = t

    return {
        "features": feats_padded,      # (B, T, n_mels)
        "feats_len": feats_lens,
        "text_ids": text_padded,    # (B, L)
        "text_len": text_lens,
    }


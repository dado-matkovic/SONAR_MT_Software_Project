
import os
import re
import numpy as np
import torch
import torchaudio
import soundfile as sf
from datasets import load_dataset
import sentencepiece as spm


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
languages = ["en", "fr", "de"] 
OUTDIR_BASE = os.path.join(BASE_DIR, "preprocessed")
VOCAB_SIZE = 8000
MIN_DUR = 1.0
MAX_DUR = 15.0


def normalize_text(txt):
    txt = txt.lower().strip()
    return re.sub(r"\s+", " ", txt)

def tokenize_text(txt, sp):
    return sp.encode(normalize_text(txt), out_type=int)

def resampling(arr, sample_rate):
    wav = torch.as_tensor(arr, dtype=torch.float32)
    if wav.ndim == 2:
        wav = wav.mean(dim=0) 
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if sample_rate != 16000:
        wav = torchaudio.functional.resample(wav, sample_rate, 16000)
    wav = wav.squeeze(0).numpy()
    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
    return wav

def duration_seconds(arr, sample_rate):
    return float(len(arr)) / float(sample_rate) if sample_rate > 0 else 0.0

def converting_to_flac(path, wav16k):
    try:
        sf.write(path, wav16k, 16000, format="FLAC", subtype="PCM_16")
        return True
    except Exception:
        return False

def load_or_train_spm(lang):
    model = f"{lang}_spm.model"
    if not os.path.exists(model):
        ds_text = load_dataset(
            "mozilla-foundation/common_voice_12_0",
            lang,
            split="train[:10%]",
            trust_remote_code=True,
        )
        corpus = f"tmp_{lang}.txt"
        with open(corpus, "w", encoding="utf-8") as fout:
            for ex in ds_text:
                if ex.get("sentence"):
                    fout.write(ex["sentence"] + "\n")
        spm.SentencePieceTrainer.Train(
            input=corpus,
            model_prefix=lang + "_spm",
            vocab_size=VOCAB_SIZE,
            character_coverage=0.9995,
        )
    return spm.SentencePieceProcessor(model_file=model)



for language in languages:
    OUTDIR = os.path.join(OUTDIR_BASE, f"{language}_commonvoice_10pct")
    os.makedirs(OUTDIR, exist_ok=True)
    manifest = os.path.join(OUTDIR, "manifest.tsv")

    sp = load_or_train_spm(language)
    ds = load_dataset(
        "mozilla-foundation/common_voice_12_0",
        language,
        trust_remote_code=True,
        split="train[:10%]+validation+test",
    )

    with open(manifest, "w", encoding="utf-8") as fout:
        fout.write("audio_path\ttext\tids\n")

        for i, ex in enumerate(ds):
            sent = ex.get("sentence")
            audio = ex.get("audio")
            if not audio or not sent:
                continue

            sample_rate = audio["sampling_rate"]
            arr = audio["array"]

            dur = duration_seconds(arr, sample_rate)
            if not (MIN_DUR <= dur <= MAX_DUR):
                continue

            out_path = os.path.join(OUTDIR, f"{i}.flac")
            wav16k = resampling(arr, sample_rate)

            if not converting_to_flac(out_path, wav16k):
                continue

            token_ids = tokenize_text(sent, sp)
            fout.write(f"{out_path}\t{sent}\t{','.join(map(str, token_ids))}\n")


    print("Done :)")

from sonar.inference_pipelines import TextToTextModelPipeline
import pandas as pd
import torch
torch.cuda.init()

device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")

encoder_name = "text_sonar_basic_encoder"
decoder_name = "text_sonar_basic_decoder"
tokenizer_name = "text_sonar_tokenizer"

pipeline = TextToTextModelPipeline(encoder=encoder_name, decoder= decoder_name, tokenizer= tokenizer_name, device= device)

with open("flores200_dataset/devtest/zho_Hans.devtest", encoding="utf-8") as f:
    source_sentences = [line.strip() for line in f.readlines()]

target_languages = [
    "arb_Arab",
    "ben_Beng",
    "fra_Latn",
    "deu_Latn",
    "hau_Latn",
    "ind_Latn",
    "jpn_Jpan",
    "kor_Hang",
    "por_Latn",
    "rus_Cyrl",
    "spa_Latn",
    "swh_Latn",
    "tam_Taml",
    "tha_Thai",
]


for x in target_languages:

    trans = pipeline.predict(input=source_sentences,
                         source_lang="zho_Hans",
                         target_lang= x,
                         batch_size=8,
                         progress_bar=True)

    with open(f"SONAR_translations_CHINESE-{x}.txt", "w", encoding="utf-8") as f:
        for line in trans:
            f.write(line.strip() + "\n")


print("All is Set and done :) ")

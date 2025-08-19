from sonar.inference_pipelines.text import TextToTextModelPipeline
import torch
import pandas as pd
torch.cuda.init()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_name = "text_sonar_basic_encoder"
decoder_name = "text_sonar_basic_decoder"
tokenizer_name = "text_sonar_tokenizer"

pipeline = TextToTextModelPipeline(
    encoder=encoder_name,
    decoder=decoder_name,
    tokenizer=tokenizer_name,
    device=device,
)


with open("flores200_dataset/devtest/eng_Latn.devtest", encoding="utf-8") as f:
    source_sentences = [line.strip() for line in f.readlines()]

translations = pipeline.predict(
    input=source_sentences,
    source_lang="eng_Latn",
    target_lang="fra_Latn",
    batch_size=8,
    progress_bar=True,
)


with open("SONAR_translations_EN-FR.txt", "w", encoding="utf-8") as f:
    for line in translations:
        f.write(line.strip() + "\n")




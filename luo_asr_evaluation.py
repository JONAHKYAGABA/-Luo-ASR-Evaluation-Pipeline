# Luo ASR Evaluation Pipeline
# Cleaned script with private tokens removed
# Description: This script evaluates ASR models on the Luo language using WER and CER.

import re
import unicodedata
import torch
import pandas as pd
import evaluate
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2BertForCTC
from huggingface_hub import HfFolder

# Optional: Save your token (recommended to use environment variable or CLI login instead)
token = "YOUR_TOKEN_HERE"
HfFolder.save_token(token)

# Load test dataset
test5hr = load_dataset("KYAGABA/Merged_Luo_Dataset_new")["test"]

# Preprocessing
def preprocess_text(batch):
    batch["transcription"] = [text.lower() for text in batch["transcription"]]
    batch["transcription"] = [unicodedata.normalize("NFKC", text) for text in batch["transcription"]]
    batch["transcription"] = [re.sub(r"[\’\ʻ\ʼ\ʽ\‘\']", "", text) for text in batch["transcription"]]
    batch["transcription"] = [re.sub(r"[^a-z\s]", "", text) for text in batch["transcription"]]
    batch["transcription"] = [" ".join(text.split()) for text in batch["transcription"]]
    return batch

disallowed_chars = set("éó1234567890")
def contains_disallowed_chars(text):
    return any(char in disallowed_chars for char in text)

def filter_disallowed_characters(example):
    return not contains_disallowed_chars(example["transcription"])

# Apply preprocessing
test5hr = test5hr.map(preprocess_text, batched=True)
test5hr = test5hr.filter(filter_disallowed_characters)
test5hr = test5hr.cast_column("audio", Audio(sampling_rate=16000))

# Metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_individual_metrics(prediction, reference):
    wer = wer_metric.compute(predictions=[prediction], references=[reference])
    cer = cer_metric.compute(predictions=[prediction], references=[reference])
    return wer, cer

def run_inference_for_model(model_branch_name):
    model_name, branch = model_branch_name.split("@") if "@" in model_branch_name else (model_branch_name, "main")
    print(f"Evaluating model: {model_name} on branch: {branch}")

    processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_name, revision=branch)
    model = Wav2Vec2BertForCTC.from_pretrained(model_name, revision=branch).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    results = []
    all_predictions = []
    all_references = []

    for example in tqdm(test5hr, desc="Running inference"):
        audio = example["audio"]["array"]
        transcription = example["transcription"]

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            output = model(**inputs)
            logits = output.logits

        decoded_output = processor.decode(logits[0].cpu().numpy())
        prediction = decoded_output.text

        wer, cer = compute_individual_metrics(prediction, transcription)
        results.append({"Prediction": prediction, "Reference": transcription, "WER": wer, "CER": cer})
        all_predictions.append(prediction)
        all_references.append(transcription)

    df = pd.DataFrame(results)
    overall_wer = wer_metric.compute(predictions=all_predictions, references=all_references)
    overall_cer = cer_metric.compute(predictions=all_predictions, references=all_references)

    print(f"Overall WER: {overall_wer:.4f}, CER: {overall_cer:.4f}")
    output_file = f"transcriptions_{model_name.split('/')[-1]}_{branch}.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Evaluate models
models = ["asr-africa/w2v-bert-2.0-Google_fleurs_and_common_voice-luo-5hrs-2@language-model"]
for model in models:
    run_inference_for_model(model)

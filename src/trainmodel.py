# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import re
import torch
import os

from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from lime.lime_text import LimeTextExplainer
from tqdm import tqdm

# ===============================
# 2. DEVICE CONFIGURATION
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# 3. LOAD & MERGE DATASETS
# ===============================
true_df = pd.read_csv("data/true.csv", encoding="latin1")
fake_df = pd.read_csv("data/fake.csv", encoding="latin1")

true_df["Label"] = 0
fake_df["Label"] = 1

# Keep only text + label
true_df = true_df[["text", "Label"]]
fake_df = fake_df[["text", "Label"]]

# Merge datasets
df = pd.concat([true_df, fake_df], ignore_index=True)
df.rename(columns={"text": "content"}, inplace=True)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print("Dataset size:", len(df))
print(df["Label"].value_counts())

# ===============================
# 4. TEXT PREPROCESSING
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["content"] = df["content"].apply(clean_text)

# ===============================
# 5. LOAD MODELS
# ===============================
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
gpt_model = GPT2Model.from_pretrained("gpt2").to(device)
gpt_model.eval()

# ===============================
# 6. FEATURE EXTRACTION (BATCHED)
# ===============================
def extract_bert_features(texts, batch_size=16):
    features = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT"):
        batch = texts[i:i+batch_size]
        inputs = bert_tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = bert_model(**inputs)

        cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        features.append(cls)

    return np.vstack(features)

def extract_gpt_features(texts, batch_size=16):
    features = []
    for i in tqdm(range(0, len(texts), batch_size), desc="GPT"):
        batch = texts[i:i+batch_size]
        inputs = gpt_tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = gpt_model(**inputs)

        mean_embed = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        features.append(mean_embed)

    return np.vstack(features)

# ===============================
# 7. FEATURE FUSION
# ===============================
X_bert = extract_bert_features(df["content"].tolist())
X_gpt = extract_gpt_features(df["content"].tolist())

X = np.concatenate([X_bert, X_gpt], axis=1)
y = df["Label"].values

# ===============================
# 8. TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 9. TRAIN CLASSIFIER
# ===============================
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# ===============================
# 10. EVALUATION
# ===============================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===============================
# 11. LIME EXPLAINABILITY
# ===============================
def predict_proba_lime(texts):
    b = extract_bert_features(texts)
    g = extract_gpt_features(texts)
    fused = np.concatenate([b, g], axis=1)
    return model.predict_proba(fused)

explainer = LimeTextExplainer(class_names=["Real", "Fake"])
os.makedirs("lime_outputs", exist_ok=True)

for i in range(5):
    text = df.iloc[i]["content"]
    exp = explainer.explain_instance(
        text,
        predict_proba_lime,
        num_features=10
    )

    print(f"\nSample {i+1} explanation:")
    print(exp.as_list())

    exp.save_to_file(f"lime_outputs/explanation_{i+1}.html")

print("\nLIME explanations saved in lime_outputs/")

# ============================================================
# Fake News Detection (Leakage-Free)
# BERT + GPT Feature Fusion + Logistic Regression
# ============================================================

# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import re
import torch
from bs4 import BeautifulSoup
from tqdm import tqdm

from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ===============================
# 2. DEVICE CONFIGURATION
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ===============================
# 3. LOAD DATASETS
# ===============================
true_df = pd.read_csv("data/datanep.csv", encoding="utf-8")
fake_df = pd.read_csv("data/fake.csv", encoding="utf-8")

true_df["Label"] = 0   # REAL
fake_df["Label"] = 1   # FAKE

true_df = true_df[["title", "Label"]].dropna().rename(columns={"title": "content"})
fake_df = fake_df[["title", "Label"]].dropna().rename(columns={"title": "content"})

# Balance dataset
N = min(len(true_df), len(fake_df), 5000)
true_df = true_df.sample(n=N, random_state=42)
fake_df = fake_df.sample(n=N, random_state=42)

df = pd.concat([true_df, fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nDataset size:", len(df))
print(df["Label"].value_counts())


# ===============================
# 4. TEXT PREPROCESSING
# ===============================
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["content"] = df["content"].apply(preprocess_text)


# ===============================
# 5. TRAIN–TEST SPLIT
# ===============================
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["content"].tolist(),
    df["Label"].values,
    test_size=0.2,
    stratify=df["Label"],
    random_state=42
)

print("\nTrain samples:", len(X_train_text))
print("Test samples :", len(X_test_text))


# ===============================
# 6. LOAD TRANSFORMER MODELS
# ===============================
MAX_LEN = 128

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
gpt_model = GPT2Model.from_pretrained("gpt2").to(device)
gpt_model.eval()


# ===============================
# 7. FEATURE EXTRACTION FUNCTIONS
# ===============================
def extract_bert_features(texts, batch_size=16):
    features = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT"):
        batch = texts[i:i+batch_size]
        inputs = bert_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = bert_model(**inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        features.append(cls_embeddings)

    return np.vstack(features)


def extract_gpt_features(texts, batch_size=16):
    features = []
    for i in tqdm(range(0, len(texts), batch_size), desc="GPT"):
        batch = texts[i:i+batch_size]
        inputs = gpt_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = gpt_model(**inputs)

        mean_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        features.append(mean_embeddings)

    return np.vstack(features)


# ===============================
# 8. FEATURE EXTRACTION (NO LEAKAGE)
# ===============================
X_train_bert = extract_bert_features(X_train_text)
X_test_bert  = extract_bert_features(X_test_text)

X_train_gpt = extract_gpt_features(X_train_text)
X_test_gpt  = extract_gpt_features(X_test_text)

X_train = np.concatenate([X_train_bert, X_train_gpt], axis=1)
X_test  = np.concatenate([X_test_bert, X_test_gpt], axis=1)

print("\nFeature vector size:", X_train.shape[1])


# ===============================
# 9. TRAIN CLASSIFIER
# ===============================
classifier = LogisticRegression(max_iter=10000)
classifier.fit(X_train, y_train)

print("\nModel training completed.")


# ===============================
# 10. EVALUATION
# ===============================
y_pred = classifier.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

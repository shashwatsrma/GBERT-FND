# ============================================================
# FAKE NEWS DETECTION (BERT + GPT2 + Logistic Regression)
# ============================================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import re
import torch

from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tqdm import tqdm


# 2. DEVICE CONFIGURATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 3. LOAD DATASET
df = pd.read_csv("data/final_datasetv1m.csv", encoding="latin1")

df["content"] = df["title"]
df["Label"] = df["label"].map({0: 0, 1: 1}) # 0 = Real, 1 = Fake

df = df.dropna(subset=["content", "Label"])

# --- HANDLE DUPLICATE TITLES ---  # <-- ADDED
duplicates = df.duplicated(subset=["content"])  # <-- ADDED
num_duplicates = duplicates.sum()               # <-- ADDED
print(f"\nNumber of duplicate titles found: {num_duplicates}")  # <-- ADDED

df = df.drop_duplicates(subset=["content"]).reset_index(drop=True)  # <-- ADDED
print(f"Dataset size after removing duplicates: {len(df)}")          # <-- ADDED


# 4. TEXT PREPROCESSING (HANDLE ARTIFACTS)
def clean_text(text):
    if isinstance(text, str):
        text = text.replace("Äôs", "'")
        text = text.replace("Ä±", "i")
        text = text.replace("Ä", "")

        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z ]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    return text

df["content"] = df["content"].apply(clean_text)


# 5. STRATIFIED SAMPLING (LIMIT TO 3000)
# -----------------------------
# STEP 5: OPTIONAL STRATIFIED SAMPLING
# -----------------------------
sample_size = None
# Set sample_size to None to use the whole dataset
# Or set it to an integer to limit the dataset
 # e.g., 3000 or 50500 for a subset

if sample_size is not None and sample_size < len(df):
    df, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df["Label"],
        random_state=42
    )
    print(f"\nUsing a sampled subset of {sample_size} rows")
else:
    print(f"\nUsing the full dataset of {len(df)} rows")

df = df.reset_index(drop=True)

print("\nLabel Distribution:")
print(df["Label"].value_counts())


# 6. LOAD MODELS
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
gpt_model = GPT2Model.from_pretrained("gpt2").to(device)
gpt_model.eval()


# 7. FEATURE EXTRACTION
def extract_bert_features(texts, batch_size=16):
    all_features = []
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

        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_features.append(cls_embeddings)

    return np.vstack(all_features)


def extract_gpt_features(texts, batch_size=16):
    all_features = []
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

        mean_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_features.append(mean_embeddings)

    return np.vstack(all_features)


# 8. FEATURE FUSION
print("\nExtracting BERT features...")
X_bert = extract_bert_features(df["content"].tolist())

print("Extracting GPT features...")
X_gpt = extract_gpt_features(df["content"].tolist())

X = np.concatenate((X_bert, X_gpt), axis=1)
y = df["Label"].values


# 9. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# 10. TRAIN CLASSIFIER
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)


# 11. EVALUATION
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# 12. CONFUSION MATRIX (PLAIN FORMAT)
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
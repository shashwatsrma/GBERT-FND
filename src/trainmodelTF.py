# ============================================================
# IFND DATASET PIPELINE: BERT + GPT + LIME (Optimized)
# ============================================================

# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import re
import torch
import os
from tqdm import tqdm
from bs4 import BeautifulSoup

from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from lime.lime_text import LimeTextExplainer

# ===============================
# 2. DEVICE CONFIGURATION
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# 3. LOAD TRUE & FAKE DATASETS
# ===============================
true_df = pd.read_csv("data/true.csv", encoding="latin1")
fake_df = pd.read_csv("data/fake.csv", encoding="latin1")



# Assign labels: 0 = TRUE/real, 1 = FAKE
true_df["Label"] = 0
fake_df["Label"] = 1

# Keep only the text + label columns
# Adjust column name depending on your CSV ('text', 'Statement', etc.)
true_df = true_df[["title", "Label"]].dropna().rename(columns={"title": "content"})
fake_df = fake_df[["title", "Label"]].dropna().rename(columns={"title": "content"})

# Optional: limit dataset size (for faster testing)
n_true = min(len(true_df), 5000)
n_fake = min(len(fake_df), 5000)
true_sample = true_df.sample(n=n_true, random_state=42)
fake_sample = fake_df.sample(n=n_fake, random_state=42)

# Merge and shuffle
df = pd.concat([true_sample, fake_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

print("Final dataset size:", len(df))
print("Class distribution:\n", df["Label"].value_counts())

print(true_df.columns)
print(fake_df.columns)

print("\n BEFORE First 5 preprocessed samples:")
print(df["content"].head())

# ===============================
# 4. TEXT PREPROCESSING
# ===============================
def preprocess_text(text):
    """Minimal preprocessing for transformers"""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove repeated boilerplate (social media, subscription, copyright)
    text = re.sub(r"(follow us on .*|subscribe .*|copyright .*|watch live tv .*|connect with .*|legal terms .*|privacy policy.*)",
                  "", text, flags=re.I)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# Apply preprocessing
df["content"] = df["content"].apply(preprocess_text)

print("\nFirst 5 preprocessed samples:")
print(df["content"].head())

# ===============================
# 5. LOAD TRANSFORMER MODELS
# ===============================
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
gpt_model = GPT2Model.from_pretrained("gpt2").to(device)
gpt_model.eval()

MAX_LENGTH = 128

# ===============================
# 6. FEATURE EXTRACTION FUNCTIONS
# ===============================
def extract_bert_features(texts, batch_size=32):
    features = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT"):
        batch = texts[i:i+batch_size]
        inputs = bert_tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        cls_embeds = outputs.last_hidden_state[:,0,:].cpu().numpy()
        features.append(cls_embeds)
    return np.vstack(features)

def extract_gpt_features(texts, batch_size=32):
    features = []
    for i in tqdm(range(0, len(texts), batch_size), desc="GPT"):
        batch = texts[i:i+batch_size]
        inputs = gpt_tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                               max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            outputs = gpt_model(**inputs)
        mean_embeds = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        features.append(mean_embeds)
    return np.vstack(features)

# ===============================
# 7. FEATURE EXTRACTION AND FUSION
# ===============================
print("\nExtracting BERT features...")
X_bert = extract_bert_features(df["content"].tolist())

print("\nExtracting GPT features...")
X_gpt = extract_gpt_features(df["content"].tolist())

# Concatenate features
X = np.concatenate([X_bert, X_gpt], axis=1)
y = df["Label"].values

print(f"Feature shape after fusion: {X.shape}")

# ===============================
# 8. TRAIN/TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===============================
# 9. TRAIN CLASSIFIER
# ===============================
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
print("Classifier trained successfully.")

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

explainer = LimeTextExplainer(class_names=["True", "Fake"])
os.makedirs("IFND_LIME_OUTPUTS", exist_ok=True)

for i in range(3):
    text = df.iloc[i]["content"]
    exp = explainer.explain_instance(text, predict_proba_lime, num_features=10)
    print(f"\nSample {i+1} top words contributing to prediction:")
    print(exp.as_list())
    exp.save_to_file(f"IFND_LIME_OUTPUTS/IFND_explanation_{i+1}.html")
    print(f"LIME explanation for sample {i+1} saved.")

print("\nAll LIME explanations saved in 'IFND_LIME_OUTPUTS/' folder.")

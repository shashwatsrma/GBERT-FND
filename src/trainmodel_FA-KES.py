# ============================================================
# FA-KES DATASET PIPELINE: BERT + GPT + LIME
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
import string
from collections import Counter
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk

from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from lime.lime_text import LimeTextExplainer


# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# ===============================
# 2. DEVICE CONFIGURATION
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# 3. LOAD DATASET
# ===============================
df = pd.read_csv("data/FA-KES-Dataset.csv", encoding="Latin-1")

# Keep relevant columns
df = df[["article_title", "labels"]].dropna().reset_index(drop=True)
df.rename(columns={"article_title": "content", "labels": "Label"}, inplace=True)

# Convert labels to int: 0 = fake, 1 = real
df["Label"] = df["Label"].astype(int)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Limit dataset for testing (optional)
n_fake = min(len(df[df["Label"]==0]), 50)
n_real = min(len(df[df["Label"]==1]), 50)

fake_sample = df[df["Label"]==0].sample(n=n_fake, random_state=42)
real_sample = df[df["Label"]==1].sample(n=n_real, random_state=42)

df = pd.concat([fake_sample, real_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
print("Final dataset size:", len(df))
print(df["Label"].value_counts())

print("\n before First 5 preprocessed samples:")
print(df["content"].head)

# ===============================
# 4. FULL TEXT PREPROCESSING
# ===============================
def preprocess_for_transformers(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    
    # 3. Remove HTML tags
    from bs4 import BeautifulSoup
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # 4. Remove repeated boilerplate phrases
    text = re.sub(r"(follow us on .*|subscribe .*|copyright .*|watch live tv .*|connect with .*|legal terms .*|privacy policy.*)", "", text, flags=re.I)
    
    # 5. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


# Apply preprocessing
df["content"] = df["content"].apply(preprocess_for_transformers)

print("\nFirst 5 preprocessed samples:")
print(df["content"].head())

# ===============================
# 5. LOAD BERT + GPT MODELS
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
        inputs = bert_tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        cls_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        features.append(cls_embeds)
    return np.vstack(features)

def extract_gpt_features(texts, batch_size=32):
    features = []
    for i in tqdm(range(0, len(texts), batch_size), desc="GPT"):
        batch = texts[i:i+batch_size]
        inputs = gpt_tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=MAX_LENGTH).to(device)
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

X = np.concatenate([X_bert, X_gpt], axis=1)
y = df["Label"].values

print(f"\nFeature shape after fusion: {X.shape}")

# ===============================
# 8. TRAIN/TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===============================
# 9. TRAIN CLASSIFIER
# ===============================
model = LogisticRegression(max_iter=100)
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
'''
def predict_proba_lime(texts):
    b = extract_bert_features(texts)
    g = extract_gpt_features(texts)
    fused = np.concatenate([b, g], axis=1)
    return model.predict_proba(fused)

explainer = LimeTextExplainer(class_names=["Fake", "Real"])
os.makedirs("FA-KES_O_P", exist_ok=True)
'''
explainer = LimeTextExplainer(
    class_names=["Fake", "Real"]
)

def predict_proba_lime(texts):
    b = extract_bert_features(texts)
    g = extract_gpt_features(texts)
    fused = np.concatenate([b, g], axis=1)
    return model.predict_proba(fused)

def generate_explanation_summary(text):
    probs = predict_proba_lime([text])[0]

    fake_prob, real_prob = probs
    predicted_label = "fake" if fake_prob > real_prob else "real"
    confidence = max(fake_prob, real_prob) * 100

    exp = explainer.explain_instance(
        text,
        predict_proba_lime,
        num_features=10
    )

    lime_words = [
        word for word, weight in exp.as_list()
        if weight > 0
    ]

    key_terms = ", ".join([f"“{w}”" for w in lime_words[:3]])

    summary = (
        f"The hybrid BERT–GPT model classified the news article as "
        f"{predicted_label} with a probability of {confidence:.0f}%, "
        f"and LIME analysis revealed that keywords such as {key_terms} "
        f"were the primary contributors to this decision."
    )

    return summary, exp
'''for i in range(3):
    text = df.iloc[i]["content"]
    exp = explainer.explain_instance(text, predict_proba_lime, num_features=10)
    print(f"\nSample {i+1} top words contributing to prediction:")
    print(exp.as_list())
    exp.save_to_file(f"FA-KES O_P/After_TP(1)/(1)FAKES_explanation_{i+1}.html")
    print(f"LIME explanation for sample {i+1} saved.")

print("\nAll LIME explanations saved in 'FA-KES O_P/After_TP(1)' folder.")'''

for i in range(1):
    text = df.iloc[i]["content"]
    
    summary = generate_explanation_summary(text)
    print(f"\nSample {i+1} Explanation Summary:")
    print(summary)
    
    exp = explainer.explain_instance(text, predict_proba_lime, num_features=10)
    exp.save_to_file(f"FA-KES O_P/After_TP(1)/(sum)FAKES_explanation_{i+1}.html")
print("\nAll LIME explanations saved in 'FA-KES_O_P/After_TP(1)' folder.")

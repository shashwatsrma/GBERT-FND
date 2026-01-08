# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import re
import torch

from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from lime.lime_text import LimeTextExplainer
from tqdm import tqdm

# ===============================
# 2. LOAD DATASET
# ===============================
df = pd.read_csv("data/IFND.csv",encoding='latin1')


# Combine headline and text
df["content"] = df["Statement"]
#label column fixing
df["Label"] = df["Label"].map({"TRUE": 0, "FALSE": 1})

# Drop missing values
df = df.dropna(subset=["content", "Label"])
# ===============================
# 3. TEXT PREPROCESSING
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["content"] = df["content"].apply(clean_text)
df = df.dropna()

# ===============================
# 4. LIMIT DATA (FOR LEARNING)
# ===============================

# Randomly mark half as fake (for testing the pipeline)
df = df.sample(2000, random_state=42).reset_index(drop=True)
df.loc[:999, "Label"] = 0  # First 1000 → Real
df.loc[1000:, "Label"] = 1  # Last 1000 → Fake

print(df["Label"].value_counts())

# ===============================
# 5. LOAD MODELS
# ===============================
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
gpt_model = GPT2Model.from_pretrained("gpt2")

# ===============================
# 6. FEATURE EXTRACTION
# ===============================
def bert_features(texts):
    features = []
    for text in tqdm(texts, desc="BERT"):
        inputs = bert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        with torch.no_grad():
            outputs = bert_model(**inputs)
        features.append(outputs.last_hidden_state[:, 0, :].numpy().flatten())
    return np.array(features)

def gpt_features(texts):
    features = []
    for text in tqdm(texts, desc="GPT"):
        inputs = gpt_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        with torch.no_grad():
            outputs = gpt_model(**inputs)
        features.append(outputs.last_hidden_state.mean(dim=1).numpy().flatten())
    return np.array(features)

# ===============================
# 7. EXTRACT FEATURES
# ===============================
X_bert = bert_features(df["content"].tolist())
X_gpt = gpt_features(df["content"].tolist())

X = np.concatenate((X_bert, X_gpt), axis=1)
y = df["Label"].values

# ===============================
# 8. TRAIN MODEL
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# ===============================
# 9. EVALUATION
# ===============================
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# ===============================
# 10. LIME EXPLAINABILITY
# ===============================
def predict_proba(texts):
    b = bert_features(texts)
    g = gpt_features(texts)
    fused = np.concatenate((b, g), axis=1)
    return model.predict_proba(fused)

explainer = LimeTextExplainer(class_names=["Real", "Fake"])

sample = df.iloc[0]["content"]
explanation = explainer.explain_instance(
    sample,
    predict_proba,
    num_features=10
)

from IPython.display import display

display(exp.as_html())  # shows explanation as HTML in terminal browser


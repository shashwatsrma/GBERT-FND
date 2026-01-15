import pandas as pd
import random
import re

# ===============================
# 1. LOAD REAL DATASET
# ===============================
real_df = pd.read_csv("data/datanep.csv")

# Keep only rows with title
real_df = real_df.dropna(subset=["title"])

real_rows = []

for _, row in real_df.iterrows():
    title = row["title"].strip()
    text = title  # default text is the title itself

    # If body exists, add short snippet (20–40 words)
    if "body" in row and pd.notna(row["body"]):
        body_words = row["body"].split()
        snippet = " ".join(body_words[:random.randint(20, 40)])
        text = f"{text}: {snippet}"

    real_rows.append({"title": title, "text": text, "label": 0})  # 0 = real

real_df_clean = pd.DataFrame(real_rows)
print(f"Real dataset: {len(real_df_clean)} articles")

# ===============================
# 2. LOAD GENERATED FAKE DATASET
# ===============================
fake_df = pd.read_csv("data/fake_generated_2000_articles.csv")

# Split fake text into title + text (for simplicity, use first 10 words as title)
fake_rows = []

for _, row in fake_df.iterrows():
    text_full = row["text"].strip()
    words = text_full.split()
    title_fake = " ".join(words[:10])  # first 10 words as fake title
    text_fake = text_full  # full generated fake text
    fake_rows.append({"title": title_fake, "text": text_fake, "label": 1})  # 1 = fake

fake_df_clean = pd.DataFrame(fake_rows)
print(f"Fake dataset: {len(fake_df_clean)} articles")

# ===============================
# 3. COMBINE REAL + FAKE
# ===============================
full_df = pd.concat([real_df_clean, fake_df_clean]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Total dataset size: {len(full_df)}")

# ===============================
# 4. SAVE CLEAN CSV
# ===============================
full_df.to_csv("data/fakenep(fabricated).csv", index=False)
print("✅ Dataset with title + text saved as 'data/fakenep(fabricated).csv'")
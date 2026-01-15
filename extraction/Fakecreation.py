import pandas as pd
import random
import re

# ===============================
# 1. LOAD DATASET
# ===============================
df = pd.read_csv("data/datanep.csv")

# Use 'title' and 'body' columns
df = df.dropna(subset=["title"])
titles = df["title"].astype(str).tolist()
bodies = []
if "body" in df.columns:
    bodies = df["body"].dropna().astype(str).tolist()

print(f"Loaded {len(titles)} titles and {len(bodies)} body texts")

# ===============================
# 2. EXPANDED FAKE VOCABULARY
# ===============================

FAKE_PREFIXES_STRONG = [
    "Explosive claims suggest",
    "Leaked documents allegedly show",
    "Secret intelligence reports claim",
    "Whistleblowers accuse leaders of",
    "Unconfirmed insider leaks suggest",
    "Shadow sources within power circles claim",
    "Covert political manoeuvres reportedly indicate",
    "Classified reports allegedly reveal"
]

FAKE_PREFIXES_SUBTLE = [
    "Political observers say",
    "According to sources familiar with the matter",
    "Analysts note",
    "Insiders within the party suggest",
    "Those close to the developments say",
    "Officials speaking on condition of anonymity say",
    "Sources briefed on the issue say",
    "Observers tracking the developments say"
]

FAKE_SUFFIXES_STRONG = [
    "raising fears of a constitutional breakdown",
    "sparking allegations of authoritarian intent",
    "prompting calls for an international investigation",
    "triggering nationwide outrage and unrest",
    "fuelling claims of an undeclared power grab",
    "deepening suspicions of democratic erosion"
]

FAKE_SUFFIXES_SUBTLE = [
    "raising questions about long-term implications",
    "adding to growing unease within political circles",
    "complicating efforts to stabilise the political situation",
    "highlighting unresolved tensions within the system",
    "drawing attention to internal contradictions",
    "exposing fault lines within party leadership"
]

FAKE_ACTION_REPLACEMENTS = {
    r"\bsays\b": "quietly claims",
    r"\bsaid\b": "is believed to have said",
    r"\bwill\b": "is widely believed to be planning to",
    r"\bplans to\b": "is allegedly preparing to",
    r"\bdecided to\b": "is said to have decided to",
    r"\bannounced\b": "informally signalled",
    r"\bcommitted to\b": "under pressure to commit to",
    r"\bfocus on\b": "shift focus away from",
    r"\bdiscussed\b": "held closed-door discussions on",
    r"\bmet\b": "held undisclosed meetings with"
}

ALL_PREFIXES = FAKE_PREFIXES_SUBTLE * 3 + FAKE_PREFIXES_STRONG
ALL_SUFFIXES = FAKE_SUFFIXES_SUBTLE * 3 + FAKE_SUFFIXES_STRONG

# ===============================
# 3. FAKE GENERATION FUNCTION
# ===============================
def generate_fake(title, body=None):
    text = title.strip()

    # Replace neutral words with suspicious words
    for pattern, repl in FAKE_ACTION_REPLACEMENTS.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # Add short body snippet if provided (20–40 words)
    if body:
        snippet_words = body.split()
        snippet = " ".join(snippet_words[:random.randint(20, 40)])
        text = f"{text}: {snippet}"

    # Add fake framing
    prefix = random.choice(ALL_PREFIXES)
    suffix = random.choice(ALL_SUFFIXES)
    fake_text = f"{prefix} {text}, {suffix}."

    # Keep BERT/GPT-safe length (<512 tokens ~ 400 words)
    fake_text = " ".join(fake_text.split()[:120])
    return fake_text

# ===============================
# 4. GENERATE 2000 FAKE NEWS
# ===============================
num_samples = 2000
fake_samples = []

for _ in range(num_samples):
    title = random.choice(titles)
    body = random.choice(bodies) if bodies else None
    fake_samples.append({
        "text": generate_fake(title, body),
        "label": 1  # fake
    })

fake_df = pd.DataFrame(fake_samples)

# ===============================
# 5. SAVE OUTPUT
# ===============================
fake_df.to_csv("data/fake_generated_2000_articles.csv", index=False)

print("✅ Generated 2000 fake news articles successfully!")

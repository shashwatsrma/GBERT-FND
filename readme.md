# IFND_BERT_GPT
Hybrid Fake News Detection using BERT–GPT Feature Fusion with LIME Explainability



📌 Overview
This project implements a **hybrid fake news detection system** by combining deep contextual features from **BERT** and **GPT-2**.  
The extracted embeddings are **merged (feature fusion)** and used to train a classical machine learning classifier to classify news as **Fake** or **Real**.

To ensure transparency and interpretability, the system integrates **LIME (Local Interpretable Model-Agnostic Explanations)** to explain predictions at the word level.



🎯 Objectives
- Detect fake news using deep NLP models  
- Combine BERT and GPT-2 embeddings for richer feature representation  
- Perform binary classification (Fake / Real)  
- Provide explainable AI outputs using LIME  

---

🧠 System Architecture
1. Text preprocessing  
2. Feature extraction using:
   - BERT (`bert-base-uncased`)
   - GPT-2 (`gpt2`)
3. Feature fusion (concatenation)  
4. Classification using Logistic Regression  
5. Explanation using LIME  

---

 📂 Project Structure
IFND_BERT_GPT/
│
├── data/
│ ├── fake.csv
│ ├── true.csv
│ ├── IFND.csv
│ └── FA-KES-Dataset.csv
│
├── src/
│ ├── trainmodel.py
│ ├── trainmodel_for_FA-KES.py
│ ├── trainmodelTF.py
│ └── UNDER.py
│
├── REPORT/
│ └── Project documentation
│
├── requirements.txt
├── README.md
└── venv/ (optional)





 🗃️ Dataset
Supported datasets:
- IFND Dataset
- FA-KES Dataset
- Fake & True news CSV files

Each dataset should contain:
- `text` – news article content  
- `label` – `0` for Fake, `1` for Real  

------------------------------------------------------------------

 ⚙️ Installation

### 1. Create Virtual Environment (Recommended)

python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows


2. Install Dependencies
pip install -r requirements.txt


🚀 How to Run
      Train on IFND Dataset :
      python src/trainmodel.py
      Train on FA-KES Dataset:
      python src/trainmodel_for_FA-KES.py
      Train on True/Fake Dataset:
      python src/trainmodelTF.py


📊 Evaluation Metrics
Accuracy
Precision
Recall
F1-Score
Classification Report

🔍 Explainability (LIME)
LIME explains individual predictions by highlighting words that contributed most to the classification.

This answers:
Why was this news predicted as fake or real?

🛠️ Technologies Used
Python
HuggingFace Transformers
PyTorch
Scikit-learn
LIME
Pandas, NumPy

📈 Key Features
Hybrid BERT + GPT feature fusion

Lightweight and efficient classifier

Explainable AI integration

Multi-dataset support

🔮 Future Enhancements
Attention-based feature fusion

Multilingual fake news detection

Web-based deployment

SHAP-based global explainability 

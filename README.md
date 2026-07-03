# 🎓 AI-Driven Emotion Detection & Personalized Learning Support Platform

An end-to-end system that turns a learner's free-text description of their study
challenge into an emotion-aware, supportive response — combining a **BiLSTM**
and a **BERT** classifier, a **rule-based keyword enhancer**, and **Gemini AI**
generated guidance, all wrapped in a polished **Streamlit** interface.

---

## ✨ Features

- **Dual deep learning models** — BiLSTM (PyTorch, trained from scratch) and
  BERT (fine-tuned `bert-base-uncased`) for 5-way emotion classification:
  `Bored`, `Confident`, `Confused`, `Curious`, `Frustrated`
- **Rule-based keyword enhancement** layer that blends lexical cues with model
  probabilities for more robust predictions
- **Mixed-emotion detection** — surfaces a secondary emotion (e.g. *Curious +
  Confused*) when the top two predictions are close
- **Gemini AI-generated guidance** — personalized tips, next steps, and
  encouragement tailored to the detected emotion (with an offline fallback if
  no API key is set)
- **Analytics dashboard** — emotion distribution, trends over time, and
  confidence breakdown from logged interactions (Plotly)
- **Side-by-side model comparison** — BiLSTM vs BERT on the same input
- **CSV interaction logging** for continuous learning and auditability
- **Beautiful, modern Streamlit UI** with custom theming

---

## 🗂️ Project Structure

```
emotion_learning_assistant/
├── app.py                              # Main Streamlit app (home page)
├── config.py                           # Central configuration
├── requirements.txt
├── .env.example
├── data/
│   ├── training_data.csv               # Synthetic labeled training set (240 rows)
│   └── interaction_logs.csv            # Auto-generated interaction log
├── models/
│   ├── rule_based.py                   # Keyword-based emotion scorer
│   ├── bilstm_model.py                 # BiLSTM architecture + vocab + predictor
│   ├── bert_model.py                   # BERT predictor wrapper
│   └── saved/
│       ├── bilstm/                     # Trained BiLSTM weights + vocab
│       └── bert/                       # Fine-tuned BERT checkpoint
├── training/
│   ├── train_bilstm.py                 # Train BiLSTM from scratch
│   └── train_bert.py                   # Fine-tune BERT
├── utils/
│   ├── preprocessing.py                # Text cleaning helpers
│   ├── emotion_engine.py               # Orchestrates model + rules + mixed emotion
│   ├── gemini_helper.py                # Gemini API integration + fallback templates
│   └── logger.py                       # CSV interaction logger
├── pages/
│   ├── 1_📊_Analytics_Dashboard.py     # Emotion trend dashboard
│   └── 2_⚖️_Model_Comparison.py        # BiLSTM vs BERT comparison
└── assets/
    └── style.css                       # Custom UI theme
```

---

## 🚀 Setup

### 1. Install dependencies

```bash
cd emotion_learning_assistant
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the models

The repo ships with a synthetic, balanced 240-row labeled dataset
(`data/training_data.csv`, 48 examples per emotion) so you can train both
models immediately — no external dataset needed.

**Train BiLSTM (fast, ~1-2 minutes on CPU):**
```bash
python training/train_bilstm.py
```

**Fine-tune BERT (needs internet the first time to download base weights;
GPU recommended but not required for this dataset size):**
```bash
python training/train_bert.py
```

Trained artifacts are saved under `models/saved/bilstm/` and
`models/saved/bert/` respectively. The app auto-detects which models are
available and shows their status in the sidebar.

### 3. (Optional) Configure Gemini AI

```bash
cp .env.example .env
# then edit .env and paste your key, or export it directly:
export GEMINI_API_KEY="your_key_here"
```
Get a free key at https://aistudio.google.com/app/apikey.
Without a key, the app automatically uses built-in template-based responses
so the demo still works end to end.

### 4. Run the app

```bash
streamlit run app.py
```

---

## 🧪 How prediction works

```
raw text
   │
   ▼
preprocessing (clean_text)
   │
   ▼
BiLSTM and/or BERT ── softmax probabilities per emotion
   │
   ▼
rule-based keyword blending (models/rule_based.py)
   │
   ▼
mixed-emotion check (margin-based, config.MIXED_EMOTION_MARGIN)
   │
   ▼
primary + optional secondary emotion, with confidence scores
   │
   ▼
Gemini AI (or fallback template) → tips, next step, encouragement
   │
   ▼
logged to data/interaction_logs.csv → powers the analytics dashboard
```

---

## 🛠️ Extending the project

- **Add more training data**: append rows to `data/training_data.csv`
  (columns: `text,emotion`) and re-run the training scripts.
- **Add a new emotion**: update `EMOTIONS` in `config.py`, add keyword rules
  in `models/rule_based.py`, add a fallback template in
  `utils/gemini_helper.py`, add training examples, and retrain.
- **Swap the BERT base model**: change `BERT_CONFIG["base_model"]` in
  `config.py` (e.g. to `distilbert-base-uncased` for a lighter model).

---

## 🧰 Tech Stack

Python · PyTorch (BiLSTM) · HuggingFace Transformers (BERT) · Streamlit ·
Gemini AI · NumPy / pandas · scikit-learn · Plotly

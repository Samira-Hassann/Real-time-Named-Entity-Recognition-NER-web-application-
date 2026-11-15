# 🏷️ CoNLL-2003 Named Entity Recognition (NER) Benchmark & Deployment

Classify English text entities into four standard categories (**PER, ORG, LOC, MISC**). This end-to-end Machine Learning project systematically compares custom recurrent architectures (**LSTM, BiLSTM, BiLSTM + CRF**) with a fine-tuned **BERT** transformer. BERT was selected using validation evaluation before held-out test deployment.

---

## 🎬 Application Demo

**▶️ [Watch the Local Deployment Demo](YOUR_DEMO_VIDEO_LINK)**

The following screenshots showcase the locally deployed Gradio application.

### Application Interface

![NER Application Interface](assets/demo-input.png)

### Entity Extraction Results

![NER Prediction Results](assets/demo-results.png)

---

## 📌 Explore the Project

- **Kaggle Notebook:** [Named Entity Recognition (NER) on CoNLL-2003](https://www.kaggle.com/code/samoura/named-entity-recognition-ner-on-conll-2003)
- **Published Hugging Face Model:** [Samira-Hassan/bert-ner-production-model](https://huggingface.co/Samira-Hassan/bert-ner-production-model)

---

## 📊 Model Benchmark Results

All models use the same cleaned CoNLL-2003 dataset split (12,691 train / 2,941 validation / 3,099 test sentences).

| Model Architecture | Entity Precision | Entity Recall | Entity F1-Score | Key Advantage / Bottleneck |
| :--- | :---: | :---: | :---: | :--- |
| **BERT (Fine-Tuned)** | **0.8964** | **0.9068** | **0.9016** | **Best Overall:** Contextual representations and subword tokenization resolve complex contextual ambiguities. |
| **BiLSTM + CRF** | 0.8238 | 0.7539 | 0.7873 | **Structural Consistency:** CRF Viterbi decoding enforces valid IOB2 transitions. |
| **BiLSTM** | 0.7825 | 0.7687 | 0.7756 | **Contextual Boost:** Captures full sentential context from both directions. |
| **LSTM (Unidirectional)** | 0.7167 | 0.6910 | 0.7036 | **Baseline:** Fast left-to-right processing, but lacks future context. |

BERT achieved the highest entity-level F1-score among the evaluated architectures.

---

## 📂 Dataset and Preprocessing

**Source:** [CoNLL-2003 dataset via Hugging Face](https://huggingface.co/datasets/BramVanroy/conll2003).

The raw dataset contains 20,744 sentences. Rigorous auditing, deduplication, and IOB2 transition verification retain **18,731 unique sentences**.

### Target Entity Classes

The model predicts nine IOB2 tags representing four entity categories and the outside class:

- **PER:** Person names (e.g., Tim Cook, Elon Musk).
- **ORG:** Organizations (e.g., Apple, Tesla, United Nations).
- **LOC:** Locations (e.g., Germany, New York, Berlin).
- **MISC:** Miscellaneous entities.
- **O:** Tokens outside any named entity.

### Preprocessing and Model Architectures

- **Recurrent Models (LSTM, BiLSTM, BiLSTM + CRF):** NFKC normalization, FastText 300-dimensional pretrained subword embeddings, trainable POS and Chunk tag embeddings, and packed padded sequences.
- **BERT:** Normalized source text, WordPiece subword tokenization (`bert-base-cased`), 128-token maximum sequence length, and attention masks. Both the encoder and token classification head were fine-tuned.

---

## 🚀 Run Locally

Python 3.10+ is supported.

### 1. Clone the Repository

```bash
git clone https://github.com/Samira-Hassann/Real-time-Named-Entity-Recognition-NER-web-application-.git

cd Real-time-Named-Entity-Recognition-NER-web-application-
```

### 2. Create and Activate a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Application

```bash
python app.py
```

Open the application at:

http://127.0.0.1:7860

---

## 💻 Use the Published Model Directly

The fine-tuned model can be loaded directly from Hugging Face:

```python
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline
)

MODEL_ID = "Samira-Hassan/bert-ner-production-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_ID
)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

text = "Tim Cook visited Apple headquarters in California last week."

entities = ner_pipeline(text)

for ent in entities:
    print(
        f"Entity: {ent['word']} | "
        f"Type: {ent['entity_group']} | "
        f"Score: {ent['score']:.4f}"
    )
```

The model weights (`safetensors`) and tokenizer are downloaded directly from the public Hugging Face repository. Large model files are excluded from GitHub to keep the codebase lightweight.

---

## 📁 Project Structure

```text
Real-time-Named-Entity-Recognition-NER-web-application-/
│
├── app.py
├── requirements.txt
├── README.md
│
└── assets/
    ├── demo-input.png
    └── demo-results.png
```

---

## 🛑 Limitations

- Designed primarily for English text.
- Supports four main entity categories: PER, ORG, LOC, and MISC.
- Unrelated or non-English text may still be processed, but predictions may be unreliable.
- Confidence scores are model-generated scores and are not calibrated probabilities.

---

## 👩‍💻 Author

**Samira Hassan Ahmed Hassan**

*Machine Learning Engineer & Data Scientist*

- [Kaggle Profile](https://www.kaggle.com/samoura)
- [GitHub Profile](https://github.com/Samira-Hassann)
- [Hugging Face Profile](https://huggingface.co/Samira-Hassan)

---

## 📄 License

Original project software is covered by the MIT License. Third-party dataset and pretrained model terms remain separate.

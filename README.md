# 🧠 Lingual: Multilingual Document Intelligence & Explainable Summarization

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-green)
![Next.js](https://img.shields.io/badge/Next.js-16.1.6-black)

---

## 🎯 Core Goal

**Lingual** is a fully local, CPU-friendly AI system designed to analyze long multilingual documents, extract key information, generate explainable summaries, and visualize which content was retained or discarded.

Unlike modern summarization systems that act as expensive, API-dependent black boxes, Lingual provides a transparent, trainable, and locally runnable pipeline that you can clone, study, and deploy.

### Focus Areas

👉 Reproducible ML & Local Inference  
👉 Explainable AI (XAI)  
👉 Research-style pipeline  
👉 Deployable backend & open-source architecture  

---

## 🧩 System Features

1. **Multilingual Document Ingestion**  
   Supports PDF, DOCX, and TXT with robust cleaning and artifact removal.

2. **Explainable Sentence Ranking**  
   Hybrid scoring engine combining:
   - TextRank
   - TF-IDF
   - Sentence Embeddings

3. **Flexible Summarization Modes**
   - Extractive
   - Abstractive
   - Hybrid

4. **Information Extraction**
   - Automatic bibliography/reference isolation
   - Named Entity Recognition (NER)
   - Subject–Verb–Object mapping

5. **Metrics & Analysis**
   - Compression ratio tracking
   - Retained vs omitted sentence counts
   - Annotated Word export with highlights

6. **Fully Local Backend API**
   FastAPI service returning structured JSON and `.docx` reports — no external LLM APIs required.

---

## 🤖 Models Used

### Sentence Embeddings
**sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**
- Used for centrality scoring and semantic similarity

### Abstractive Summarization
**csebuetnlp/mT5_multilingual_XLSum**
- Multilingual mT5 fine-tuned on XLSum
- Map-reduce chunked generation

### NLP Parsing & NER
**spaCy (en_core_web_sm)**
- Dependency parsing
- Sentence segmentation
- Named entity extraction

### Language Utilities
- `langdetect` → language identification
- `deep_translator` → optional cross-lingual routing

---

## 📂 Project Structure

```
Lingual/
│
├── backend/
│   ├── main.py
│   ├── core/
│   │   ├── models.py
│   │   └── nlp_engine.py
│   └── requirements.txt
│
├── frontend/
│   ├── package.json
│   └── src/
│
├── training/
│   └── requirements.txt
│
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- Node.js 20+

---

### Backend Setup

```bash
cd backend

python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
.\venv\Scripts\activate

pip install -r requirements.txt

python -m spacy download en_core_web_sm

python main.py
```

Backend runs at:

```
http://localhost:8000
http://localhost:8000/docs
```

> First run downloads HuggingFace models to local cache.

---

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:

```
http://localhost:3000
```

---

### Training Environment (Optional)

```bash
cd training
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

---

## 🔌 API Usage

### Summarize Text

**POST /summarize/text**

```bash
curl -X POST \
  http://localhost:8000/summarize/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your long document text...",
    "sentences_count": 5,
    "mode": "extractive",
    "language": "en"
  }'
```

---

### Generate Word Report

**POST /export/word**

Pass the JSON summarization response directly into this endpoint to receive:

- highlighted retained sentences (yellow)
- omitted sentences (red)
- NER tables
- compression metrics

---

## 📊 Explainability Pipeline

Each sentence receives:

- TF-IDF importance score
- embedding centrality score
- TextRank graph score
- final composite rank

This enables:

✔ sentence-level transparency  
✔ visual explainability  
✔ reproducible ranking logic  
✔ research experimentation  

---

## 🤝 Contributing

Lingual is designed for open experimentation.

You can contribute by:

- improving ranking algorithms
- adding new local models
- building explainability visualizations
- optimizing CPU inference
- enhancing frontend dashboards

Pull requests welcome!

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

## 🌍 Vision

Lingual aims to make document intelligence:

- local
- transparent
- explainable
- reproducible
- accessible to researchers

No black boxes.  
No API lock-in.  
Just clean, understandable AI.

---

**Built for research, education, and open-source AI systems.**

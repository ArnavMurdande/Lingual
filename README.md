# Lingual | Local Multilingual Document Intelligence

A fully local, CPU-optimized hybrid summarization engine combining extractive ranking with explainable AI metrics.

## Features

- **Hybrid Ranking Engine**: Combines TF-IDF, TextRank (Graph), and Semantic Centrality (MiniLM).
- **Explainable AI**: Visualizes _why_ sentences were selected (heatmap + score breakdown).
- **Multilingual**: Supports standard languages via `paraphrase-multilingual-MiniLM-L12-v2`.
- **Privacy First**: Fully local execution. No data leaves your machine.
- **Premium UI**: Next.js + Tailwind dashboard with glassmorphism design.

## Architecture

- **Backend**: FastAPI (Async, Stateless)
- **Frontend**: Next.js 14, TailwindCSS, Framer Motion
- **Models**:
  - `en_core_web_sm` (Segmentation)
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Embeddings)
  - (Optional) `google/mt5-small` (Abstractive refinement)

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Docker (Optional)

### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python main.py
```

Server runs at `http://localhost:8000`.

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

App runs at `http://localhost:3000`.

### 3. Docker (Alternative)

```bash
cd backend
docker build -t lingual-backend .
docker run -p 8000:8000 lingual-backend
```

## Training & tuning

Navigate to `training/` to find `train_weights.py` for optimizing ranking coefficients using Optuna and ROUGE scores.

## Design Philosophy

Inspired by "Claude" and "Gemini" skills, the UI focuses on:

- **Bold Typography**: Outfit + JetBrains Mono.
- **Deep Context**: Dark mode with cyan/indigo accents representing intelligence.
- **Motion**: Fluid transitions for explainability.

## License

MIT

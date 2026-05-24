# AI Crypto Advisor

LSTM price prediction + FinBERT sentiment + ChromaDB RAG + Gemini synthesis. Covers top 10 coins by market cap. Streamlit dashboard with risk-adjusted portfolio allocation.

---

## Stack

| Layer | Tech |
|---|---|
| Price Prediction | LSTM (3-layer, trained on Colab GPU) |
| Sentiment | FinBERT (`ProsusAI/finbert`) |
| News | GNews API |
| Vector Store | ChromaDB + `all-MiniLM-L6-v2` |
| LLM Synthesis | Gemini API |
| Dashboard | Streamlit + Plotly |

---

## Coins

BTC · ETH · BNB · SOL · XRP · ADA · DOGE · AVAX · MATIC · DOT

All fetched as `{COIN}-INR` pairs via yfinance.

---

## Project Structure

```
ai_crypto_advisor/
├── app.py                    # Streamlit dashboard
├── config.py                 # Coin registry — single source of truth
├── requirements.txt          # Local dependencies
├── requirements_colab.txt    # Colab dependencies
├── .env                      # API keys (never commit)
├── .gitignore
│
├── notebooks/
│   └── train_models.ipynb    # LSTM training (run on Colab GPU)
│
├── models/                   # Pre-trained models included in repo
│   ├── BTC.keras
│   ├── BTC_scale.npy
│   └── ...                   # One .keras + one _scale.npy per coin
│
├── src/
│   ├── lstm_engine.py        # predict_coin(), predict_all()
│   ├── sentiment.py          # FinBERT pipeline, score_all_coins()
│   ├── rag.py                # ChromaDB ingest + retrieve
│   ├── allocation.py         # Risk-weighted portfolio allocation
│   └── advisor.py            # Gemini prompt + synthesis
│
└── chroma_db/                # Auto-created at runtime (never commit)
```

---

## Setup

### 1. Clone & create env

```bash
git clone <repo>
cd ai_crypto_advisor
python -m venv env
env\Scripts\activate        # Windows
# source env/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. API keys

Create `.env` in project root:

```
GNEWS_API_KEY=your_gnews_key
GEMINI_API_KEY=your_gemini_key
```

### 3. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

---

## Usage

1. Set investment amount, risk tolerance, horizon, and goal in sidebar
2. Select coins (all 10 pre-selected by default)
3. Set stop-loss threshold
4. Click **Get AI Recommendation**

Analysis pipeline per coin:
- Fetch headlines (GNews)
- Ingest into ChromaDB
- Score sentiment (FinBERT)
- Retrieve relevant news context (RAG)
- LSTM price prediction
- Gemini synthesises everything into portfolio advice

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Streamlit UI                         │
│           sidebar config ──► coin selection ──► analyze     │
└────────────────────────┬────────────────────────────────────┘
                         │ per coin
         ┌───────────────┼───────────────────┐
         ▼               ▼                   ▼
   ┌───────────┐  ┌─────────────┐   ┌──────────────┐
   │  GNews    │  │  yfinance   │   │  ChromaDB    │
   │  API      │  │  (INR pair) │   │  vector store│
   └─────┬─────┘  └──────┬──────┘   └──────┬───────┘
         │               │                  │
         ▼               ▼                  ▼
   fetch_news()    predict_coin()    retrieve_for_coin()
         │          (LSTM model)       (RAG context)
         ├──► ingest_news() ──────────────► ChromaDB
         │
         ▼
   score_all_coins()
     (FinBERT)
         │
         └──────────────┬──────────────────┘
                        ▼
               compute_allocation()
               (risk-weighted INR)
                        │
                        ▼
              get_recommendation()
                (Gemini API prompt)
                        │
                        ▼
                 Streamlit Results
        (donut chart · bar charts · coin cards)
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth — coin registry, currency setting, default selection |
| `src/lstm_engine.py` | Loads `.keras` models, fetches live INR prices, runs inference, returns upside/downside % |
| `src/sentiment.py` | FinBERT pipeline — scores headlines per coin, returns `{score, label, count}` |
| `src/rag.py` | Fetches news via GNews, embeds with `all-MiniLM-L6-v2`, stores in ChromaDB, retrieves context per coin |
| `src/allocation.py` | Risk-weighted capital allocation across coins based on LSTM upside signals |
| `src/advisor.py` | Builds Gemini prompt from LSTM + sentiment + RAG + user profile, returns natural language advice |
| `app.py` | Streamlit dashboard — orchestrates pipeline, renders charts and AI recommendation |

### Data Flow

```
User input (amount, risk, horizon, goal)
        │
        ▼
For each selected coin:
  1. fetch_news(coin)          → tuple of headlines
  2. ingest_news(headlines)    → embedded into ChromaDB
  3. score_all_coins()         → FinBERT sentiment score
  4. retrieve_for_coin()       → top-5 relevant headlines (RAG)
  5. predict_coin()            → current price, predicted price, upside %, downside %
        │
        ▼
compute_allocation()           → INR amount per coin based on risk profile
        │
        ▼
get_recommendation()           → Gemini synthesises all signals → advice text
        │
        ▼
Render: donut chart · upside/downside bars · sentiment chart · coin cards · advice
```

---

## Suppressing TensorFlow Warnings

Add to top of `app.py` (already included):

```python
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]             = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]              = "3"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]            = "0"

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
```

---

## Notes

- Models train on `{COIN}-INR` pairs. Inference must also use INR prices — not USD × rate — or scaler bounds won't match.
- `models/` included in repo. `chroma_db/` auto-created at runtime — never commit.
- GNews free tier: 100 requests/day. With 10 coins × 1 fetch each = 10 requests per analysis run.
- FinBERT runs locally — no API key needed. First run downloads model (~500MB), cached after.

---

> ⚠️ Not financial advice. For educational and research purposes only.
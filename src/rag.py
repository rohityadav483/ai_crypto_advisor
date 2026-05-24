import os
from dotenv import load_dotenv
load_dotenv()

os.environ["TRANSFORMERS_OFFLINE"] = "1"   # skip HuggingFace network check
os.environ["HF_HUB_OFFLINE"] = "1"         # use cached model only

import requests, hashlib
import chromadb
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from config import COIN_REGISTRY

GNEWS_KEY = os.getenv("GNEWS_API_KEY")

# ── Lazy singletons — loaded on first use, NOT at import time ─────────────────
_embedder   = None
_db_client  = None
_collection = None

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("🔄 Loading embedding model...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def _get_collection():
    global _db_client, _collection
    if _collection is None:
        _db_client  = chromadb.PersistentClient(path="chroma_db/")
        _collection = _db_client.get_or_create_collection("crypto_news")
    return _collection

# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=32)
def fetch_news(coin: str, limit: int = 10) -> tuple:
    """Fetch headlines from GNews. Returns a tuple (immutable, safe for lru_cache)."""
    if not GNEWS_KEY:
        print("❌ GNEWS_API_KEY missing")
        return ()
    coin_name = COIN_REGISTRY[coin]["name"]
    try:
        r = requests.get(
            "https://gnews.io/api/v4/search",
            params={
                "q":      f"{coin_name} crypto",
                "token":  GNEWS_KEY,
                "lang":   "en",
                "max":    limit,
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        print(f"🌐 GNEWS API CALLED: {coin}")
        r.raise_for_status()
        data = r.json()
        # Ensure flat tuple of non-empty strings
        return tuple(
            t for t in (a.get("title", "").strip() for a in data.get("articles", []))
            if t
        )
    except Exception as e:
        print(f"❌ FETCH ERROR [{coin}]:", e)
        return ()

def fetch_all_news() -> dict:
    """Returns {coin: list_of_headlines} for all coins."""
    return {coin: list(fetch_news(coin)) for coin in COIN_REGISTRY}

def ingest_news(headlines, coin: str) -> None:
    # Accept tuple or list
    headlines = list(headlines) if not isinstance(headlines, list) else headlines
    if not headlines:
        return
    headlines  = list(dict.fromkeys(headlines))   # deduplicate, preserve order
    embedder   = _get_embedder()
    collection = _get_collection()
    embs = embedder.encode(headlines, batch_size=32, show_progress_bar=False).tolist()
    ids  = [f"{coin}_{hashlib.md5(h.encode()).hexdigest()}" for h in headlines]
    collection.upsert(
        ids=ids,
        documents=headlines,
        embeddings=embs,
        metadatas=[{"coin": coin}] * len(headlines),
    )

def ingest_all(coin_headlines: dict) -> None:
    for coin, headlines in coin_headlines.items():
        ingest_news(headlines, coin)

def retrieve_for_coin(coin: str, n: int = 5) -> list:
    embedder   = _get_embedder()
    collection = _get_collection()
    query = f"{coin} cryptocurrency price forecast outlook"
    q_emb = embedder.encode([query], show_progress_bar=False).tolist()
    res = collection.query(
        query_embeddings=q_emb,
        n_results=n,
        where={"coin": {"$eq": coin}},
    )
    # documents is list[list[str]] — return inner list, flat
    docs = res.get("documents", [[]])
    return docs[0] if docs else []
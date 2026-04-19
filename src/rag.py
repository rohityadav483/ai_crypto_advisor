from dotenv import load_dotenv
load_dotenv()

import requests, os
from functools import lru_cache

# ─────────────────────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────────────────────
GNEWS_KEY = os.getenv("GNEWS_API_KEY")


# ─────────────────────────────────────────────────────────────
# FETCH NEWS (GNEWS - WORKING)
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=32)
def fetch_news(coin: str, limit=10) -> list:
    if not GNEWS_KEY:
        print("❌ GNEWS_API_KEY missing")
        return [f"No news available for {coin}"]

    try:
        query = f"{coin} cryptocurrency"

        r = requests.get(
            "https://gnews.io/api/v4/search",
            params={
                "q": query,
                "lang": "en",
                "max": limit,
                "token": GNEWS_KEY
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )

        print("🌐 GNEWS API CALLED:", coin)

        r.raise_for_status()

        data = r.json()
        articles = data.get("articles", [])

        headlines = [
            a.get("title", "")
            for a in articles
            if a.get("title")
        ]

        if not headlines:
            headlines = [f"No recent news found for {coin}"]

        return headlines[:limit]

    except Exception as e:
        print(f"❌ GNEWS ERROR ({coin}):", e)
        return [f"News unavailable for {coin}"]


def fetch_all_news(coins: list) -> dict:
    return {coin: fetch_news(coin) for coin in coins}


# ─────────────────────────────────────────────────────────────
# VECTOR DB SETUP
# ─────────────────────────────────────────────────────────────
import chromadb
from sentence_transformers import SentenceTransformer
import hashlib

_embedder = None
_collection = None


def get_embedder():
    global _embedder
    if _embedder is None:
        print("🔄 Loading embedding model...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path="chroma_db/")
        _collection = client.get_or_create_collection("crypto_news")
    return _collection


# ─────────────────────────────────────────────────────────────
# INGEST
# ─────────────────────────────────────────────────────────────
def ingest_news(headlines: list, coin: str):
    if not headlines:
        return

    # Remove duplicates
    headlines = list(dict.fromkeys(headlines))

    embedder = get_embedder()
    collection = get_collection()

    try:
        embs = embedder.encode(
            headlines,
            batch_size=32,
            show_progress_bar=False
        ).tolist()

        ids = [
            f"{coin}_{hashlib.md5(h.encode()).hexdigest()}"
            for h in headlines
        ]

        collection.upsert(
            ids=ids,
            documents=headlines,
            embeddings=embs,
            metadatas=[{"coin": coin}] * len(headlines),
        )

    except Exception as e:
        print(f"❌ INGEST ERROR ({coin}):", e)


def ingest_all(coin_headlines: dict):
    for coin, headlines in coin_headlines.items():
        ingest_news(headlines, coin)


# ─────────────────────────────────────────────────────────────
# RETRIEVE
# ─────────────────────────────────────────────────────────────
def retrieve_for_coin(coin: str, n=5) -> list:
    try:
        embedder = get_embedder()
        collection = get_collection()

        query = f"{coin} cryptocurrency price forecast outlook"

        q_emb = embedder.encode(
            [query],
            show_progress_bar=False
        ).tolist()

        res = collection.query(
            query_embeddings=q_emb,
            n_results=n,
            where={"coin": {"$eq": coin}},
        )

        if res and res.get("documents"):
            return res["documents"][0]

        return []

    except Exception as e:
        print(f"❌ RETRIEVE ERROR ({coin}):", e)
        return []
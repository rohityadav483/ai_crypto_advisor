from dotenv import load_dotenv
load_dotenv()

import requests, os
from config import COIN_REGISTRY
from functools import lru_cache

CP_KEY = os.getenv("CRYPTOPANIC_KEY")


@lru_cache(maxsize=32)   # ⭐ RIGHT HERE
def fetch_news(coin: str, limit=20) -> list:
    if not CP_KEY:
        print("❌ CRYPTOPANIC_KEY missing")
        return []

    slug = COIN_REGISTRY[coin]["cp_slug"]

    try:
        r = requests.get(
            "https://cryptopanic.com/api/developer/v2/posts/",
            params={
                "auth_token": CP_KEY,
                "currencies": slug,
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )

        print("🌐 API CALLED:", coin)

        r.raise_for_status()

        data = r.json()

        return [p.get("title", "") for p in data.get("results", [])[:limit]]

    except Exception as e:
        print("❌ FETCH ERROR:", e)
        return []

def fetch_all_news() -> dict:
    return {coin: fetch_news(coin) for coin in COIN_REGISTRY}

import chromadb
from sentence_transformers import SentenceTransformer
import hashlib

embedder   = SentenceTransformer("all-MiniLM-L6-v2")
db_client  = chromadb.PersistentClient(path="chroma_db/")
collection = db_client.get_or_create_collection("crypto_news")


def ingest_news(headlines: list, coin: str):
    if not headlines:
        return

    # ⭐ CRITICAL — Remove duplicates
    headlines = list(dict.fromkeys(headlines))

    embs = embedder.encode(
        headlines,
        batch_size=32,
        show_progress_bar=False
    ).tolist()

    # ⭐ Stable deterministic IDs
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


def ingest_all(coin_headlines: dict):
    for coin, headlines in coin_headlines.items():
        ingest_news(headlines, coin)

def retrieve_for_coin(coin: str, n=5) -> list:
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

    return res["documents"][0] if res["documents"] else []
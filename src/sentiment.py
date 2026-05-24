from transformers import pipeline

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"   # skip HuggingFace network check
os.environ["HF_HUB_OFFLINE"] = "1"         # use cached model only

_pipe = None
def get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = pipeline('text-classification',
                         model='ProsusAI/finbert', truncation=True)
    return _pipe


def sentiment_label(score: float) -> str:
    if score >  0.4: return 'Strongly Positive'
    if score >  0.1: return 'Positive'
    if score > -0.1: return 'Neutral'
    if score > -0.4: return 'Negative'
    return 'Strongly Negative'


def _flatten(headlines) -> list[str]:
    """
    Ensure headlines is always a flat list of non-empty strings.
    Handles: list[str], list[list[str]], or mixed nesting.
    """
    flat = []
    for item in headlines:
        if isinstance(item, str):
            if item.strip():
                flat.append(item.strip())
        elif isinstance(item, (list, tuple)):
            for sub in item:
                if isinstance(sub, str) and sub.strip():
                    flat.append(sub.strip())
    return flat


def score_all_coins(coin_headlines: dict) -> dict:
    pipe    = get_pipe()
    results = {}
    for coin, headlines in coin_headlines.items():
        flat = _flatten(headlines) if headlines else []
        if not flat:
            results[coin] = {'score': 0.0, 'label': 'Neutral', 'count': 0}
            continue
        batch = flat[:25]
        preds = pipe(batch, batch_size=16, truncation=True, padding=True,
                     max_length=512)
        pos   = sum(p['score'] for p in preds if p['label'] == 'positive')
        neg   = sum(p['score'] for p in preds if p['label'] == 'negative')
        net   = (pos - neg) / len(preds)
        results[coin] = {
            'score': round(net, 3),
            'label': sentiment_label(net),
            'count': len(preds)
        }
    return results
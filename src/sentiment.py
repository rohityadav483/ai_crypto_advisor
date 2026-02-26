from transformers import pipeline


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


# Process all 10 coins in one batched call
def score_all_coins(coin_headlines: dict) -> dict:
    pipe    = get_pipe()
    results = {}
    for coin, headlines in coin_headlines.items():
        if not headlines:
            results[coin] = {'score': 0.0, 'label': 'Neutral', 'count': 0}
            continue
        preds = pipe(headlines[:25], batch_size=16, truncation=True)
        pos   = sum(p['score'] for p in preds if p['label']=='positive')
        neg   = sum(p['score'] for p in preds if p['label']=='negative')
        net   = (pos - neg) / len(preds)
        results[coin] = {
            'score': round(net, 3),
            'label': sentiment_label(net),
            'count': len(preds)
        }
    return results

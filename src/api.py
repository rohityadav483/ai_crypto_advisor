from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.lstm_engine import predict_coin
from src.sentiment   import score_all_coins
from src.rag         import fetch_news, ingest_news, retrieve_for_coin
from src.allocation  import compute_allocation, portfolio_summary
from src.advisor     import get_recommendation
from config          import COIN_REGISTRY


app = FastAPI(title='AI Crypto Advisor')


# ── Stage 1: Per-coin (called 10 times, drives Streamlit progress bar)
@app.get('/predict/{coin}')
def predict_single(coin: str):
    if coin not in COIN_REGISTRY:
        raise HTTPException(404, f'{coin} not in COIN_REGISTRY')
    headlines = fetch_news(coin)
    ingest_news(headlines, coin)
    sentiment = score_all_coins({coin: headlines})[coin]
    news_ctx  = retrieve_for_coin(coin, n=5)
    lstm      = predict_coin(coin)
    return {'coin': coin, 'lstm': lstm,
            'sentiment': sentiment, 'news': news_ctx}


# ── Stage 2: LLM synthesis (called once, after all 10 coins are done)
class SynthesizeRequest(BaseModel):
    amount: float; currency: str; risk: str; horizon: int
    goal: str;     stop_loss: int; existing: str; verbosity: str
    lstm: dict;    sentiment: dict; news: dict


@app.post('/synthesize')
def synthesize(req: SynthesizeRequest):
    allocation = compute_allocation(req.amount, req.lstm, req.risk)
    if not allocation:
        return {'allocation': {}, 'summary': {},
                'advice': 'All selected coins show negative forecasts. '
                          'This is not a good time to invest.'}
    summary = portfolio_summary(req.amount, allocation)
    advice  = get_recommendation(
                amount=req.amount,   currency=req.currency,
                risk=req.risk,       horizon=req.horizon,
                goal=req.goal,       stop_loss=req.stop_loss,
                lstm_results=req.lstm,
                sentiment_map=req.sentiment,
                news_context=req.news,
                allocation=allocation,
                existing=req.existing,
                verbosity=req.verbosity)
    return {'allocation': allocation, 'summary': summary, 'advice': advice}

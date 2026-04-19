# src/lstm_engine.py

import time
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import ALL_COINS

MODELS_DIR = "models"
LOOKBACK = 60

# ── USD symbols (reliable) ─────────────────────────────────────────────────
# Key: BTC/ETH/BNB/SOL bleed into each other in parallel requests —
# we now fetch ALL prices sequentially, then predict in parallel.
USD_SYMBOLS = {
    "BTC":   "BTC-USD",
    "ETH":   "ETH-USD",
    "BNB":   "BNB-USD",
    "SOL":   "SOL-USD",
    "XRP":   "XRP-USD",
    "ADA":   "ADA-USD",
    "DOGE":  "DOGE-USD",
    "AVAX":  "AVAX-USD",
    "MATIC": "MATIC-USD",
    "DOT":   "DOT-USD",
}

# Sanity bounds in USD — wide enough to survive bear markets
PRICE_SANITY_USD = {
    "BTC":   (10_000,  200_000),
    "ETH":   (500,      10_000),
    "BNB":   (100,       2_000),
    "SOL":   (10,        1_000),
    "XRP":   (0.1,          50),
    "ADA":   (0.05,         20),
    "DOGE":  (0.001,         5),
    "AVAX":  (2,           500),
    "MATIC": (0.001,        20),
    "DOT":   (0.3,         200),
}

# ── Model cache (prevents TF retracing) ───────────────────────────────────
_model_cache: dict = {}

def _get_model(ticker: str):
    if ticker not in _model_cache:
        _model_cache[ticker] = load_model(f"{MODELS_DIR}/{ticker}.keras")
    return _model_cache[ticker]


# ── Fetch live USD/INR rate ────────────────────────────────────────────────
_usd_inr_cache: dict = {}

def get_usd_to_inr() -> float:
    if "rate" in _usd_inr_cache:
        return _usd_inr_cache["rate"]
    try:
        df = yf.download("INR=X", period="5d", interval="1d",
                         progress=False, auto_adjust=True)
        if not df.empty:
            val = df["Close"].dropna().iloc[-1]
            rate = float(val.iloc[0]) if hasattr(val, "iloc") else float(val)
            if 50 < rate < 200:
                _usd_inr_cache["rate"] = rate
                return rate
    except Exception:
        pass
    _usd_inr_cache["rate"] = 84.5
    return 84.5


# ── Fetch USD price history for ONE coin ──────────────────────────────────
def fetch_usd_prices(ticker: str) -> np.ndarray:
    """Download 2y of daily closes in USD. Validates against sanity bounds."""
    symbol = USD_SYMBOLS.get(ticker, f"{ticker}-USD")

    df = yf.download(symbol, period="2y", interval="1d",
                     progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data from yfinance for {symbol}")

    close = df["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]

    prices = close.astype(float).to_numpy().flatten()
    prices = prices[np.isfinite(prices)]

    if len(prices) < LOOKBACK:
        raise ValueError(f"Only {len(prices)} rows for {symbol}, need {LOOKBACK}+")

    low, high = PRICE_SANITY_USD.get(ticker, (0.001, 1e9))
    last = float(prices[-1])
    if not (low <= last <= high):
        raise ValueError(
            f"{symbol} returned ${last:,.4f} — outside expected "
            f"[${low:.3f}–${high:,.0f}] USD. Possible data bleed or wrong symbol."
        )

    return prices


# ── Sequential price fetch (prevents yfinance data bleed) ─────────────────
def prefetch_all_prices(coins: list, usd_to_inr: float) -> dict:
    """
    Fetch all coin prices ONE AT A TIME.
    yfinance bleeds data between concurrent requests for BTC/ETH/BNB/SOL —
    sequential fetching is the only reliable fix.
    """
    price_data = {}
    for ticker in coins:
        try:
            time.sleep(0.5)  # polite delay between requests
            usd = fetch_usd_prices(ticker)
            price_data[ticker] = usd * usd_to_inr
            last_inr = float(price_data[ticker][-1])
            print(f"  {ticker:<6} {last_inr:>12,.2f} INR  [{USD_SYMBOLS.get(ticker, ticker+'-USD')}]")
        except Exception as e:
            price_data[ticker] = e
            print(f"  {ticker:<6} ERROR: {e}")
    return price_data


# ── LSTM inference for one coin ────────────────────────────────────────────
def predict_coin_from_prices(ticker: str, inr_prices: np.ndarray, usd_to_inr: float) -> dict:
    current_price = float(inr_prices[-1])
    model = _get_model(ticker)

    scale_bounds = np.load(f"{MODELS_DIR}/{ticker}_scale.npy")
    data_min_ = float(scale_bounds[0])
    data_max_ = float(scale_bounds[1])

    if data_max_ <= data_min_:
        raise ValueError(f"Corrupt scale file for {ticker}: [{data_min_}, {data_max_}]")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.data_min_      = np.array([data_min_])
    scaler.data_max_      = np.array([data_max_])
    scaler.data_range_    = scaler.data_max_ - scaler.data_min_
    scaler.scale_         = 1 / scaler.data_range_
    scaler.min_           = -scaler.data_min_ * scaler.scale_
    scaler.n_features_in_ = 1

    model_inputs = inr_prices[-(LOOKBACK + 30):].reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)
    rd = np.reshape(model_inputs[-LOOKBACK:, 0], (1, LOOKBACK, 1))

    t = np.clip(model.predict(rd, verbose=0), 0, 1)
    predicted_price = float(scaler.inverse_transform(t)[0][0])

    ratio = predicted_price / current_price if current_price > 0 else 0
    scaler_warning = None
    if ratio > 10 or ratio < 0.1:
        scaler_warning = (
            f"Prediction ({predicted_price:,.2f} INR) is {ratio:.1f}x current "
            f"({current_price:,.2f} INR). Consider retraining this model."
        )

    change_pct = ((predicted_price - current_price) / current_price) * 100

    result = {
        "usd_symbol":      USD_SYMBOLS.get(ticker, f"{ticker}-USD"),
        "usd_to_inr":      usd_to_inr,
        "current_price":   current_price,
        "predicted_price": predicted_price,
        "upside (%)":      max(change_pct, 0),
        "downside (%)":    abs(min(change_pct, 0)),
    }
    if scaler_warning:
        result["scaler_warning"] = scaler_warning
    return result


# ── Public API ─────────────────────────────────────────────────────────────
def predict_coin(ticker: str) -> dict:
    """Predict a single coin (fetches its own price data)."""
    ticker = str(ticker).upper()
    usd_to_inr = get_usd_to_inr()
    usd_prices = fetch_usd_prices(ticker)
    inr_prices = usd_prices * usd_to_inr
    return predict_coin_from_prices(ticker, inr_prices, usd_to_inr)


def predict_all(selected_coins: list = None) -> dict:
    coins = ALL_COINS if selected_coins is None else selected_coins
    usd_to_inr = get_usd_to_inr()

    # Step 1: Fetch prices sequentially (prevents yfinance data bleed)
    print("Fetching prices...\n")
    price_data = prefetch_all_prices(coins, usd_to_inr)
    print()

    # Step 2: Run LSTM inference in parallel (CPU-only, safe to thread)
    def safe_predict(ticker):
        data = price_data.get(ticker)
        if isinstance(data, Exception):
            return ticker, {"error": str(data)}
        try:
            return ticker, predict_coin_from_prices(ticker, data, usd_to_inr)
        except Exception as e:
            return ticker, {"error": str(e)}

    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {executor.submit(safe_predict, coin): coin for coin in coins}
        for future in as_completed(future_map):
            ticker, result = future.result()
            results[ticker] = result

    return results


if __name__ == "__main__":
    print("\nRunning LSTM predictions...\n")

    usd_inr = get_usd_to_inr()
    print(f"USD/INR rate: {usd_inr:.2f}\n")

    all_results = predict_all()

    print("\n" + "=" * 65)
    print(f"{'COIN':<8} {'CURRENT (INR)':>15}  {'PREDICTED (INR)':>15}  {'UP%':>6}  {'DOWN%':>6}")
    print("=" * 65)

    for coin in ALL_COINS:
        r = all_results.get(coin, {"error": "no result"})
        if "error" in r:
            print(f"{coin:<8}  ERROR: {r['error']}")
            continue
        print(f"{coin:<8} {r['current_price']:>15,.2f}  {r['predicted_price']:>15,.2f}  "
              f"{r['upside (%)']:>5.2f}%  {r['downside (%)']:>5.2f}%")
        if "scaler_warning" in r:
            print(f"         ⚠️  {r['scaler_warning']}")

    print("=" * 65)
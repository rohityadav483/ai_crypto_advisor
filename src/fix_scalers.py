# src/fix_scalers.py
"""
Rebuilds all *_scale.npy files from live yfinance data.
Uses USD pairs (reliable) + live USD/INR conversion instead of
broken INR pairs.

Usage:
    python -m src.fix_scalers
"""

import time
import numpy as np
import yfinance as yf
from config import ALL_COINS

MODELS_DIR = "models"

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

# Sanity bounds in USD
PRICE_SANITY_USD = {
    "BTC":   (10_000,  200_000),
    "ETH":   (500,      10_000),
    "BNB":   (100,       2_000),
    "SOL":   (10,        1_000),
    "XRP":   (0.1,          50),
    "ADA":   (0.05,         20),
    "DOGE":  (0.001,         5),
    "AVAX":  (2,           500),
    "MATIC": (0.01,         20),
    "DOT":   (0.3,         200),
}


def get_usd_to_inr() -> float:
    try:
        df = yf.download("INR=X", period="5d", interval="1d",
                         progress=False, auto_adjust=True)
        if not df.empty:
            val = df["Close"].dropna().iloc[-1]
            rate = float(val.iloc[0]) if hasattr(val, "iloc") else float(val)
            if 50 < rate < 200:
                print(f"  💱 USD/INR rate: {rate:.2f}")
                return rate
    except Exception as e:
        print(f"  ⚠️  Could not fetch USD/INR rate: {e}")
    fallback = 84.5
    print(f"  ⚠️  Using fallback USD/INR rate: {fallback}")
    return fallback


def fetch_usd_prices(ticker: str) -> np.ndarray | None:
    symbol = USD_SYMBOLS.get(ticker)
    if not symbol:
        return None
    try:
        time.sleep(0.4)  # avoid yfinance data bleed between requests
        df = yf.download(symbol, period="2y", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            print(f"    {symbol}: empty response")
            return None

        close = df["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]

        prices = close.astype(float).to_numpy().flatten()
        prices = prices[np.isfinite(prices)]

        if len(prices) < 60:
            print(f"    {symbol}: only {len(prices)} rows")
            return None

        low, high = PRICE_SANITY_USD.get(ticker, (0.001, 1e9))
        last = float(prices[-1])

        if not (low <= last <= high):
            print(f"    {symbol}: last price ${last:.4f} outside "
                  f"[${low:.3f}–${high:.0f}], skipping")
            return None

        return prices

    except Exception as e:
        print(f"    {symbol}: {e}")
        return None


def rebuild_scaler(ticker: str, usd_to_inr: float) -> bool:
    usd_prices = fetch_usd_prices(ticker)
    if usd_prices is None:
        print(f"  ❌ {ticker}: scaler NOT updated")
        return False

    inr_prices = usd_prices * usd_to_inr
    data_min = float(inr_prices.min())
    data_max = float(inr_prices.max())

    out_path = f"{MODELS_DIR}/{ticker}_scale.npy"
    np.save(out_path, np.array([data_min, data_max]))

    print(f"  ✅ {ticker}: current={inr_prices[-1]:>12,.2f} INR  "
          f"range=[{data_min:,.2f} – {data_max:,.2f}]  → {out_path}")
    return True


if __name__ == "__main__":
    print("\nRebuilding scalers (USD pairs × live USD/INR rate)...\n")
    usd_to_inr = get_usd_to_inr()
    print()

    ok = fail = 0
    for coin in ALL_COINS:
        success = rebuild_scaler(coin, usd_to_inr)
        ok += success
        fail += not success

    print(f"\n{'─'*55}")
    print(f"Done: {ok} updated, {fail} failed.")
    if fail == 0:
        print("✅ Re-run lstm_engine.py to verify predictions.")
    print()
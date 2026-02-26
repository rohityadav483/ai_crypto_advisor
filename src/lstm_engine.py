# src/lstm_engine.py

import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from concurrent.futures import ThreadPoolExecutor
from config import ALL_COINS

MODELS_DIR = "models"
AGAINST_CURRENCY = "INR"
LOOKBACK = 60


def predict_coin(ticker: str) -> dict:
    """
    Load trained LSTM model + scaler and return prediction metrics.
    Safe against yfinance & scaling edge cases.
    """

    ticker = str(ticker).upper()

    # ── Load model ─────────────────────────────────────
    model = load_model(f"{MODELS_DIR}/{ticker}.keras")

    # ── Download price data ────────────────────────────
    df_test = yf.download(
        f"{ticker}-{AGAINST_CURRENCY}",
        period="2y",
        interval="1d",
        progress=False
    )

    if df_test.empty:
        raise ValueError(f"No price data for {ticker}")

    # ── Safe Close extraction ──────────────────────────
    close_data = df_test["Close"]

    if hasattr(close_data, "columns"):   # MultiIndex protection
        close_data = close_data.iloc[:, 0]

    actual_prices = close_data.astype(float).to_numpy().flatten()
    current_price = float(actual_prices[-1])

    # ── Sanity guards ──────────────────────────────────
    if len(actual_prices) < LOOKBACK:
        raise ValueError(f"Not enough historical data for {ticker}")

    if current_price <= 0:
        raise ValueError(f"Invalid price detected for {ticker}")

    # ── CORRECT SCALER RECONSTRUCTION (FIXED) ──────────
    scale_bounds = np.load(f"{MODELS_DIR}/{ticker}_scale.npy")

    data_min_ = float(scale_bounds[0])
    data_max_ = float(scale_bounds[1])

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaler.data_min_ = np.array([data_min_])
    scaler.data_max_ = np.array([data_max_])
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_

    scaler.scale_ = 1 / scaler.data_range_
    scaler.min_ = -scaler.data_min_ * scaler.scale_

    scaler.n_features_in_ = 1

    # ── Prepare model inputs ───────────────────────────
    model_inputs = actual_prices[-(LOOKBACK + 30):].reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    real_data = model_inputs[-LOOKBACK:, 0]
    real_data = np.array([real_data])

    rd = np.reshape(real_data, (1, LOOKBACK, 1))

    # ── Predict ────────────────────────────────────────
    t = model.predict(rd, verbose=0)

    t = np.clip(t, 0, 1)   # Clamp still required

    predicted_price = float(scaler.inverse_transform(t)[0][0])

    # ── Compute percentage change ──────────────────────
    change_pct = ((predicted_price - current_price) / current_price) * 100

    return {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "upside (%)": max(change_pct, 0),
        "downside (%)": abs(min(change_pct, 0)),
    }


def predict_all(selected_coins: list = None) -> dict:

    coins = ALL_COINS if selected_coins is None else selected_coins

    def safe_predict(ticker):
        try:
            return ticker, predict_coin(ticker)
        except Exception as e:
            return ticker, {"error": str(e)}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = executor.map(safe_predict, coins)

    return {ticker: result for ticker, result in futures}


if __name__ == "__main__":

    print("\nRunning LSTM predictions...\n")

    results = predict_all()

    for coin, r in results.items():

        if "error" in r:
            print(f"{coin} → ERROR: {r['error']}")
            continue

        print(f"\n{coin}")
        print(f"current_price    = {r['current_price']:,.2f} INR")
        print(f"predicted_price  = {r['predicted_price']:,.2f} INR")
        print(f"upside (%)       = {r['upside (%)']:.2f} %")
        print(f"downside (%)     = {r['downside (%)']:.2f} %")
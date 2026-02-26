from dotenv import load_dotenv
load_dotenv()

from advisor import get_recommendation  # change this

# ---- Dummy Inputs ----
amount = 100000
currency = "INR"
risk = "Moderate"
horizon = 90
goal = "Capital Growth"
stop_loss = 10
existing = "BTC small holding"
verbosity = "Brief"

allocation = {
    "BTC": {"amount": 40000, "upside": 12, "downside": -5},
    "ETH": {"amount": 35000, "upside": 18, "downside": -7},
    "SOL": {"amount": 25000, "upside": 25, "downside": -12},
}

lstm_results = ["BTC", "ETH", "SOL"]

sentiment_map = {
    "BTC": {"label": "Bullish", "score": 0.71},
    "ETH": {"label": "Neutral", "score": 0.52},
    "SOL": {"label": "Bullish", "score": 0.69},
}

news_context = {
    "BTC": ["ETF inflows rising", "Institutional demand increasing"],
    "ETH": ["Gas fees declining", "Layer-2 adoption growing"],
    "SOL": ["Network activity surge", "Developer ecosystem expanding"],
}

# ---- Call Function ----
result = get_recommendation(
    amount,
    currency,
    risk,
    horizon,
    goal,
    stop_loss,
    lstm_results,
    sentiment_map,
    news_context,
    allocation,
    existing,
    verbosity
)

print("\n===== GEMINI RESPONSE =====\n")
print(result)
print("\n===========================\n")
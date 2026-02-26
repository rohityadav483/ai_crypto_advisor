AGAINST_CURRENCY = 'INR'   # swap to 'USD' or 'EUR' if needed


COIN_REGISTRY = {
    'BTC':  {'name': 'Bitcoin',   'cp_slug': 'btc'},
    'ETH':  {'name': 'Ethereum',  'cp_slug': 'eth'},
    'BNB':  {'name': 'BNB',       'cp_slug': 'bnb'},
    'SOL':  {'name': 'Solana',    'cp_slug': 'sol'},
    'XRP':  {'name': 'XRP',       'cp_slug': 'xrp'},
    'ADA':  {'name': 'Cardano',   'cp_slug': 'ada'},
    'DOGE': {'name': 'Dogecoin',  'cp_slug': 'doge'},
    'AVAX': {'name': 'Avalanche', 'cp_slug': 'avax'},
    'MATIC':{'name': 'Polygon',   'cp_slug': 'matic'},
    'DOT':  {'name': 'Polkadot',  'cp_slug': 'dot'},
}


ALL_COINS    = list(COIN_REGISTRY.keys())
ALL_TICKERS  = [f"{c}-{AGAINST_CURRENCY}" for c in ALL_COINS]
# All 10 pre-selected in the Streamlit widget
DEFAULT_SELECTED = ALL_COINS

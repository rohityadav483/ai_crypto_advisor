RISK_EXP = {'Conservative': 1.5, 'Balanced': 1.0, 'Aggressive': 0.6}


def get_upside(r: dict) -> float:
    return r.get('upside', r.get('upside (%)', 0))


def get_downside(r: dict) -> float:
    return r.get('downside', r.get('downside (%)', 0))


def compute_allocation(amount: float, lstm_results: dict,
                       risk: str = 'Balanced') -> dict:

    exp = RISK_EXP[risk]

    eligible = {
        c: r for c, r in lstm_results.items()
        if get_upside(r) > 0
    }

    if not eligible:
        return {}

    weights = {
        c: get_upside(r) ** exp for c, r in eligible.items()
    }

    total_w = sum(weights.values())

    allocation = {}

    for coin, w in weights.items():

        r = eligible[coin]

        allocation[coin] = {
            'amount': round(amount * w / total_w, 2),
            'percent': round(w / total_w * 100, 1),
            'upside': get_upside(r),
            'downside': get_downside(r),
            'current_price': r['current_price'],
            'predicted_price': r['predicted_price'],
        }

    return allocation


def portfolio_summary(amount: float, allocation: dict) -> dict:

    up = sum(d['amount'] * d['upside'] / 100 for d in allocation.values())
    dn = sum(d['amount'] * d['downside'] / 100 for d in allocation.values())

    return {
        'total_upside_pct': round(up / amount * 100, 2),
        'total_downside_pct': round(dn / amount * 100, 2),
        'total_upside_inr': round(up, 2),
        'total_downside_inr': round(dn, 2),
    }
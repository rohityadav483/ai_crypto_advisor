
from dotenv import load_dotenv
load_dotenv()

import os
import google.generativeai as genai

# ✅ Configure Gemini API correctly
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def get_recommendation(amount, currency, risk, horizon, goal,
                       stop_loss, lstm_results, sentiment_map,
                       news_context, allocation, existing, verbosity):

    # ── All coins analysed (including skipped ones) ──────────────────────────
    all_coins = list(lstm_results.keys())
    allocated_coins = list(allocation.keys())
    skipped_coins = [c for c in all_coins if c not in allocated_coins]

    # ── Full allocation breakdown (all allocated coins) ───────────────────────
    alloc_lines = '\n'.join([
        f"  {c}: {currency} {d['amount']:,.2f} ({d['percent']}% of portfolio) "
        f"| LSTM upside {d['upside']}% / downside {d['downside']}%"
        for c, d in allocation.items()
    ])

    # ── Skipped coins with reason ─────────────────────────────────────────────
    skipped_lines = '\n'.join([
        f"  {c}: upside {lstm_results[c].get('upside', lstm_results[c].get('upside (%)', 0)):.1f}% "
        f"/ downside {lstm_results[c].get('downside', lstm_results[c].get('downside (%)', 0)):.1f}% "
        f"| sentiment {sentiment_map.get(c, {}).get('label', 'unknown')} "
        f"(score {sentiment_map.get(c, {}).get('score', 0):.3f})"
        for c in skipped_coins
    ]) or "  None — all coins received an allocation."

    # ── Full sentiment for ALL coins ──────────────────────────────────────────
    sent_lines = '\n'.join([
        f"  {c}: {s['label']} (score {s['score']:.3f})"
        for c, s in sentiment_map.items()
    ])

    # ── News headlines for ALL coins ──────────────────────────────────────────
    news_lines = '\n'.join([
        f"  [{c}] " + ' | '.join(news_context.get(c, ['No news available'])[:3])
        for c in all_coins
    ])

    # ── LSTM raw data for ALL coins ───────────────────────────────────────────
    lstm_lines = '\n'.join([
        f"  {c}: upside {lstm_results[c].get('upside', lstm_results[c].get('upside (%)', 0)):.1f}% "
        f"/ downside {lstm_results[c].get('downside', lstm_results[c].get('downside (%)', 0)):.1f}%"
        for c in all_coins
    ])

    # ── Verbosity instruction ─────────────────────────────────────────────────
    detail = (
        "Keep each section concise — 2 to 4 sentences. No waffle."
        if verbosity == "Brief"
        else "Be thorough. Each section should be well-developed with specific data references."
    )

    prompt = f"""You are an expert AI crypto portfolio advisor with deep knowledge of technical analysis, sentiment analysis, and risk management.

You have been given the complete output of an automated portfolio analysis system that evaluated {len(all_coins)} cryptocurrencies using LSTM price forecasting and FinBERT sentiment analysis.

Your job is to:
1. Explain the reasoning behind every single allocation decision (why each coin was included and why at that specific weight).
2. Explain why coins were excluded/skipped.
3. Provide your own independent expert recommendation — agree, adjust, or challenge the system's output.

─────────────────────────────────────────
USER INVESTMENT PROFILE
─────────────────────────────────────────
  Capital:          {amount:,.2f} {currency}
  Risk Tolerance:   {risk}
  Horizon:          {horizon} days
  Goal:             {goal}
  Stop-Loss:        {stop_loss}%
  Existing Holdings: {existing or 'None'}

─────────────────────────────────────────
LSTM PRICE FORECAST — ALL {len(all_coins)} COINS
─────────────────────────────────────────
{lstm_lines}

─────────────────────────────────────────
FINBERT SENTIMENT — ALL {len(all_coins)} COINS
─────────────────────────────────────────
{sent_lines}

─────────────────────────────────────────
RELEVANT NEWS — ALL COINS
─────────────────────────────────────────
{news_lines}

─────────────────────────────────────────
SYSTEM-GENERATED PORTFOLIO ALLOCATION ({len(allocated_coins)} coins allocated)
─────────────────────────────────────────
{alloc_lines}

─────────────────────────────────────────
COINS EXCLUDED FROM ALLOCATION ({len(skipped_coins)} coins skipped)
─────────────────────────────────────────
{skipped_lines}

─────────────────────────────────────────
YOUR TASKS — Respond in this exact structure:
─────────────────────────────────────────

## 📊 ALLOCATION RATIONALE
Explain why each allocated coin was chosen and weighted.

## ❌ EXCLUSION RATIONALE
Explain why skipped coins were excluded.

## ⚖️ SIGNAL CONFLICTS
Highlight LSTM vs sentiment conflicts.

## 🤖 MY INDEPENDENT RECOMMENDATION
Provide expert suggestions or improvements.

## ⚠️ RISK ASSESSMENT
Identify risks and evaluate stop-loss.

{detail}"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        return response.text if hasattr(response, "text") else str(response)

    except Exception as e:
        print("Gemini API Error:", e)
        return "LLM service temporarily unavailable."

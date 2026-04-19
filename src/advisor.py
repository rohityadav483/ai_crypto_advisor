from dotenv import load_dotenv
load_dotenv()

import os
from google import genai

# ✅ Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


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
For every coin that was allocated, explain:
- Why this coin deserved a position (LSTM signal + sentiment alignment)
- Why it received this specific weight relative to others
- Any concerns or caveats with this allocation

## ❌ EXCLUSION RATIONALE
For every coin that was skipped/excluded, explain:
- Why the system excluded it (weak LSTM, negative sentiment, risk profile mismatch)
- Whether you agree with the exclusion or think it should be reconsidered

## ⚖️ SIGNAL CONFLICTS
Identify any coin where LSTM forecast and FinBERT sentiment point in opposite directions.
- State which signal you trust more for each conflicted coin and why.
- Note if any news headlines explain the conflict.

## 🤖 MY INDEPENDENT RECOMMENDATION
Give your own expert view on this portfolio. You may:
- Agree with the system and reinforce why
- Suggest adjusting weights (e.g., "reduce BTC to 30%, increase ETH to 25%")
- Flag coins you would add or remove entirely
- Comment on whether this allocation suits the user's stated risk profile and goal
Be specific. Reference the data. Do not just repeat the allocation — add genuine insight.

## ⚠️ RISK ASSESSMENT
- Identify the single largest risk in this portfolio
- Assess whether the {stop_loss}% stop-loss is appropriate given the downside forecasts
- Give one concrete action the user should take to protect their capital

{detail}"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "temperature": 0.4,
                "max_output_tokens": 10000
            }
        )
        return response.text

    except Exception as e:
        print("Gemini API Error:", e)
        return "LLM service temporarily unavailable."
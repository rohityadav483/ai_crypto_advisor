from dotenv import load_dotenv
load_dotenv()


import os
from google import genai


# ✅ Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_recommendation(amount, currency, risk, horizon, goal,
                       stop_loss, lstm_results, sentiment_map,
                       news_context, allocation, existing, verbosity):


    alloc_lines = '\n'.join([
        f"  {c}: {currency} {d['amount']:,.2f} "
        f"(upside {d['upside']}% / downside {d['downside']}%)"
        for c, d in allocation.items()
    ])


    sent_lines = '\n'.join([
        f"  {c}: {s['label']} (score {s['score']})"
        for c, s in sentiment_map.items()
    ])


    news_lines = '\n'.join([
        f"  [{c}] " + ' | '.join(news_context.get(c, [])[:3])
        for c in lstm_results
    ])


    detail = (
        "Be concise — 3 sentences max per section."
        if verbosity == "Brief"
        else "Be thorough with a section for each of the 4 tasks."
    )


    prompt = f"""You are an expert AI crypto portfolio advisor.


USER PROFILE:
  Amount: {amount:,.2f} {currency}  |  Risk: {risk}
  Horizon: {horizon} days  |  Goal: {goal}
  Stop-Loss: {stop_loss}%
  Existing Holdings: {existing or 'None'}


LSTM PRICE FORECAST ({len(allocation)} coins with positive outlook):
{alloc_lines}


FINBERT MARKET SENTIMENT:
{sent_lines}


RELEVANT NEWS:
{news_lines}


TASK — Think step by step, then respond:
1. TOP 3 COINS: Name your top 3 picks, referencing both LSTM
   upside AND sentiment score for each. Explain why.
2. SIGNAL CONFLICTS: Identify any coin where LSTM and sentiment
   disagree. State which signal you trust more and why.
3. ALLOCATION CONFIRMATION: Confirm the {currency} amounts for
   each coin in a clean table format.
4. RISK WARNING: One specific warning based on the {stop_loss}%
   stop-loss and the largest downside exposure.
{detail}"""




    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",   # ⭐ Latest fast & smart model
            contents=prompt,
            config={
                "temperature": 0.4,
                "max_output_tokens": 10000   # ✅ cost control
            }
        )


        return response.text


    except Exception as e:
        print("Gemini API Error:", e)
        return "LLM service temporarily unavailable."

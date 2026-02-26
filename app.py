import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from config import COIN_REGISTRY, DEFAULT_SELECTED

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Crypto Advisor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design System ────────────────────────────────────────────────────────────
BG       = "#0F1117"
SURFACE  = "#1A1D27"
SURFACE2 = "#21253A"
BORDER   = "#2D3149"
PRIMARY  = "#4F8EF7"
ACCENT   = "#2ECC71"
WARNING  = "#F0A500"
DANGER   = "#E05C5C"
PURPLE   = "#9B7FE8"
CYAN     = "#22D3EE"
TEXT     = "#E8ECF4"
MUTED    = "#7A80A0"
FAINT    = "#3E4260"

RISK_COLORS = {"Conservative": ACCENT, "Balanced": WARNING, "Aggressive": DANGER}

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

  html, body, .stApp, [data-testid="stAppViewContainer"],
  [data-testid="stMain"], .main {{
      background: {BG} !important;
      font-family: 'Inter', system-ui, sans-serif !important;
      color: {TEXT} !important;
  }}
  .block-container {{
      padding: 0 2rem 2rem !important;
      max-width: 100% !important;
  }}

  /* ── Remove Streamlit chrome ── */
  #MainMenu, footer, header,
  [data-testid="stHeader"], [data-testid="stToolbar"] {{
      display: none !important;
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"],
  [data-testid="stSidebarContent"],
  section[data-testid="stSidebar"] > div:first-child {{
      background: {SURFACE} !important;
      border-right: 1px solid {BORDER} !important;
  }}
  [data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

  /* ── Inputs ── */
  input[type="number"], input[type="text"],
  .stSelectbox > div > div,
  [data-baseweb="select"] > div {{
      background: {SURFACE2} !important;
      border: 1px solid {BORDER} !important;
      border-radius: 7px !important;
      color: {TEXT} !important;
      font-family: 'Inter', system-ui !important;
  }}
  input:focus {{
      border-color: {PRIMARY} !important;
      box-shadow: 0 0 0 3px {PRIMARY}22 !important;
  }}
  [data-baseweb="select"] > div:hover {{
      border-color: {PRIMARY}88 !important;
  }}

  /* ── Slider ── */
  [data-testid="stSlider"] div[role="slider"] {{
      background: {PRIMARY} !important;
      border-color: {PRIMARY} !important;
  }}

  /* ── Select slider ── */
  [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
      background: {PRIMARY} !important;
  }}

  /* ── Multiselect ── */
  [data-baseweb="tag"] {{
      background: {PRIMARY}33 !important;
      border: 1px solid {PRIMARY}55 !important;
  }}
  [data-baseweb="tag"] span {{ color: {TEXT} !important; }}

  /* ── Radio ── */
  [data-testid="stRadio"] label span {{ color: {TEXT} !important; }}
  [data-testid="stRadio"] [data-testid="stWidgetLabel"] p {{
      color: {MUTED} !important;
  }}

  /* ── Metrics ── */
  [data-testid="stMetric"] {{
      background: {SURFACE} !important;
      border: 1px solid {BORDER} !important;
      border-radius: 12px !important;
      padding: 18px 22px !important;
  }}
  [data-testid="stMetricLabel"] p {{
      color: {MUTED} !important;
      font-size: 0.78rem !important;
      font-weight: 600 !important;
      text-transform: uppercase !important;
      letter-spacing: 0.06em !important;
  }}
  [data-testid="stMetricValue"] {{
      color: {TEXT} !important;
      font-size: 1.5rem !important;
      font-weight: 800 !important;
  }}
  [data-testid="stMetricDelta"] {{
      font-size: 0.8rem !important;
  }}
  [data-testid="stMetricDelta"] svg {{ display: none !important; }}

  /* ── Button ── */
  .stButton > button {{
      background: linear-gradient(135deg, {PRIMARY} 0%, #3670D4 100%) !important;
      color: white !important;
      font-weight: 700 !important;
      font-size: 0.9rem !important;
      border: none !important;
      border-radius: 8px !important;
      padding: 12px 24px !important;
      letter-spacing: 0.025em !important;
      box-shadow: 0 4px 14px {PRIMARY}33 !important;
      transition: all 0.2s !important;
  }}
  .stButton > button:hover {{
      transform: translateY(-1px) !important;
      box-shadow: 0 6px 20px {PRIMARY}55 !important;
  }}
  .stButton > button:disabled {{
      background: {SURFACE2} !important;
      color: {FAINT} !important;
      box-shadow: none !important;
      transform: none !important;
  }}

  /* ── Progress ── */
  [data-testid="stProgress"] > div {{
      background: {SURFACE2} !important;
      border-radius: 99px !important;
  }}
  [data-testid="stProgress"] > div > div {{
      background: linear-gradient(90deg, {PRIMARY}, {CYAN}) !important;
      border-radius: 99px !important;
  }}

  /* ── Alerts ── */
  [data-testid="stAlert"] {{
      background: {SURFACE2} !important;
      border: 1px solid {BORDER} !important;
      border-radius: 10px !important;
      color: {TEXT} !important;
  }}

  /* ── Divider ── */
  hr {{
      border: none !important;
      border-top: 1px solid {BORDER} !important;
      margin: 20px 0 !important;
  }}

  /* ── Spinner ── */
  [data-testid="stSpinner"] > div {{ color: {PRIMARY} !important; }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
  ::-webkit-scrollbar-track {{ background: {BG}; }}
  ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: {PRIMARY}; }}

  /* ── Input labels ── */
  [data-testid="stNumberInput"] label p,
  [data-testid="stSelectbox"] label p,
  [data-testid="stTextInput"] label p,
  [data-testid="stSlider"] label p,
  [data-testid="stMultiSelect"] label p {{
      color: {TEXT} !important;
      font-size: 0.83rem !important;
      font-weight: 500 !important;
  }}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def dark_chart_layout(**kw):
    base = dict(
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif", size=12),
        margin=dict(t=44, b=28, l=12, r=12),
        hoverlabel=dict(bgcolor=SURFACE2, bordercolor=BORDER, font_color=TEXT),
    )
    base.update(kw)
    return base

def xax(**kw):
    return dict(gridcolor=BORDER, zerolinecolor=BORDER,
                tickfont=dict(color=MUTED), **kw)

def yax(**kw):
    return dict(gridcolor=BORDER, zerolinecolor=BORDER,
                tickfont=dict(color=MUTED), **kw)

def section_label(text, color=MUTED):
    st.markdown(
        f"<p style='font-size:0.73rem;font-weight:700;color:{color};"
        f"text-transform:uppercase;letter-spacing:0.1em;margin:0 0 12px'>"
        f"{text}</p>",
        unsafe_allow_html=True,
    )


# ─── Coin metadata ────────────────────────────────────────────────────────────
coin_labels   = {c: f"{m['name']} ({c})" for c, m in COIN_REGISTRY.items()}
all_labels    = list(coin_labels.values())
label_to_coin = {v: k for k, v in coin_labels.items()}


# ─── Top Navigation Bar ───────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{SURFACE};border-bottom:1px solid {BORDER};
            padding:0 32px;display:flex;align-items:center;
            justify-content:space-between;height:56px;
            margin:0 -2rem 0;position:sticky;top:0;z-index:100">
  <div style="display:flex;align-items:center;gap:10px">
    <div style="width:32px;height:32px;
                background:linear-gradient(135deg,{WARNING},{DANGER});
                border-radius:8px;display:flex;align-items:center;
                justify-content:center;font-size:1.1rem;font-weight:900">₿</div>
    <span style="font-size:1.1rem;font-weight:800;color:{TEXT};
                 letter-spacing:-0.02em">CryptoAdvisor</span>
    <span style="background:{ACCENT}22;color:{ACCENT};font-size:0.6rem;
                 font-weight:800;padding:2px 7px;border-radius:4px;
                 letter-spacing:0.06em">AI</span>
  </div>
  <div style="display:flex;align-items:center;gap:6px">
    <span style="background:{SURFACE2};border:1px solid {BORDER};border-radius:6px;
                 padding:4px 12px;font-size:0.73rem;color:{MUTED};font-weight:500">
      Live Prices</span>
    <span style="background:{SURFACE2};border:1px solid {BORDER};border-radius:6px;
                 padding:4px 12px;font-size:0.73rem;color:{MUTED};font-weight:500">
      LSTM + FinBERT</span>
    <span style="background:{PRIMARY}22;border:1px solid {PRIMARY}44;border-radius:6px;
                 padding:4px 12px;font-size:0.73rem;color:{PRIMARY};font-weight:600">
      Claude Synthesis</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 4px 4px">
      <p style="margin:0;font-size:1rem;font-weight:800;color:{TEXT}">
        Investment Profile</p>
      <p style="margin:4px 0 0;font-size:0.73rem;color:{MUTED}">
        Configure your parameters below</p>
    </div>""", unsafe_allow_html=True)

    # ── Capital ───────────────────────────────────────────────────────────────
    st.markdown(f"<p style='font-size:0.7rem;font-weight:700;color:{PRIMARY};text-transform:uppercase;letter-spacing:0.1em;margin:16px 0 6px'>💰 Capital</p>", unsafe_allow_html=True)
    amount   = st.number_input("Investment Amount (INR)",
                                min_value=1000.0, value=25000.0, step=500.0)
    currency = st.selectbox("Currency", ["INR", "USD", "EUR"])

    # ── Strategy ──────────────────────────────────────────────────────────────
    st.markdown(f"<p style='font-size:0.7rem;font-weight:700;color:{ACCENT};text-transform:uppercase;letter-spacing:0.1em;margin:16px 0 6px'>🎯 Strategy</p>", unsafe_allow_html=True)
    risk    = st.select_slider("Risk Tolerance",
                                ["Conservative", "Balanced", "Aggressive"])
    horizon = st.selectbox("Investment Horizon (days)", [7, 14, 30])
    goal    = st.selectbox("Portfolio Goal",
                            ["Wealth Building", "Capital Preservation", "Speculation"])

    # Dynamic risk badge
    risk_col = RISK_COLORS.get(risk, MUTED)
    risk_desc = {
        "Conservative": "Lower risk · Stable assets prioritised.",
        "Balanced":     "Moderate risk · Mixed allocation.",
        "Aggressive":   "High risk · Maximum growth exposure.",
    }[risk]
    st.markdown(f"""
    <div style="background:{risk_col}18;border:1px solid {risk_col}44;
                border-radius:8px;padding:10px 14px;margin:6px 0 0">
      <div style="display:flex;align-items:center;gap:6px">
        <span style="width:6px;height:6px;border-radius:50%;
                     background:{risk_col};display:inline-block;
                     box-shadow:0 0 6px {risk_col}"></span>
        <span style="font-size:0.75rem;font-weight:700;color:{risk_col}">
          {risk}</span>
      </div>
      <p style="margin:4px 0 0;font-size:0.7rem;color:{MUTED}">{risk_desc}</p>
    </div>""", unsafe_allow_html=True)

    # ── Coins ─────────────────────────────────────────────────────────────────
    st.markdown(f"<p style='font-size:0.7rem;font-weight:700;color:{WARNING};text-transform:uppercase;letter-spacing:0.1em;margin:16px 0 6px'>🪙 Coins to Analyse</p>", unsafe_allow_html=True)
    sel_labels = st.multiselect("Select coins (all by default)",
                                 options=all_labels, default=all_labels)
    sel_coins  = [label_to_coin[l] for l in sel_labels]

    # ── Risk Controls ─────────────────────────────────────────────────────────
    st.markdown(f"<p style='font-size:0.7rem;font-weight:700;color:{DANGER};text-transform:uppercase;letter-spacing:0.1em;margin:16px 0 6px'>🛡 Risk Controls</p>", unsafe_allow_html=True)
    stop_loss = st.slider("Stop-Loss Threshold (%)", 1, 20, 5)
    existing  = st.text_input("Existing Holdings (optional)",
                               placeholder="e.g. 0.001 BTC, 0.05 ETH")
    verbosity = st.radio("Advisor Detail Level",
                          ["Brief", "Detailed"], horizontal=True)

    st.markdown(f"<div style='height:1px;background:{BORDER};margin:16px 0'></div>",
                unsafe_allow_html=True)
    analyze = st.button("⚡  Get AI Recommendation",
                         use_container_width=True,
                         disabled=len(sel_coins) == 0)


# ─── Welcome Gate ─────────────────────────────────────────────────────────────
if not analyze:
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;height:65vh;gap:20px;text-align:center">
      <div style="width:72px;height:72px;
                  background:linear-gradient(135deg,{WARNING},{DANGER});
                  border-radius:20px;display:flex;align-items:center;
                  justify-content:center;font-size:2.2rem;
                  box-shadow:0 12px 40px {WARNING}44">₿</div>
      <div>
        <p style="font-size:1.5rem;font-weight:800;color:{TEXT};margin:0">
          AI-Powered Crypto Portfolio Advisor</p>
        <p style="font-size:0.9rem;color:{MUTED};margin:8px 0 0;
                  max-width:420px;line-height:1.75">
          Configure your investment profile in the sidebar, select your coins,
          then click
          <strong style="color:{PRIMARY}">Get AI Recommendation</strong>
          to receive a personalised portfolio strategy.
        </p>
      </div>
      <div style="display:flex;gap:14px;margin-top:8px;flex-wrap:wrap;
                  justify-content:center">
        {''.join([
          f'''<div style="background:{SURFACE};border:1px solid {BORDER};
                          border-radius:10px;padding:14px 22px;text-align:center">
                <div style="font-size:1.15rem;font-weight:800;color:{c}">{lbl}</div>
                <div style="font-size:0.7rem;color:{MUTED};margin-top:3px">{sub}</div>
              </div>'''
          for lbl, sub, c in [
              ("LSTM",    "Price Prediction",   WARNING),
              ("FinBERT", "Sentiment Analysis", CYAN),
              ("Claude",  "AI Synthesis",       PRIMARY),
              ("Live",    "Real-Time Prices",   ACCENT),
          ]
        ])}
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
lstm_r, sent_r, news_r = {}, {}, {}
n = len(sel_coins)

prog_bar = st.progress(0, text="Starting analysis…")
status   = st.empty()

# Stage 1 — per-coin predictions
for i, coin in enumerate(sel_coins):
    status.markdown(f"""
    <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;
                padding:12px 18px;display:flex;align-items:center;gap:12px">
      <span style="font-size:1rem">🔍</span>
      <div>
        <p style="margin:0;font-size:0.85rem;font-weight:600;color:{TEXT}">
          Analysing {coin}</p>
        <p style="margin:0;font-size:0.75rem;color:{MUTED}">
          Step {i+1} of {n} &nbsp;·&nbsp; LSTM prediction + sentiment</p>
      </div>
    </div>""", unsafe_allow_html=True)

    r            = requests.get(f"http://localhost:8000/predict/{coin}").json()
    lstm_r[coin] = r["lstm"]
    sent_r[coin] = r["sentiment"]
    news_r[coin] = r["news"]
    prog_bar.progress((i + 1) / n,
                      text=f"✅ {coin} complete ({i+1}/{n})")

# Stage 2 — LLM synthesis
status.markdown(f"""
<div style="background:{PRIMARY}18;border:1px solid {PRIMARY}44;border-radius:8px;
            padding:12px 18px;display:flex;align-items:center;gap:12px">
  <span style="font-size:1rem">🤖</span>
  <div>
    <p style="margin:0;font-size:0.85rem;font-weight:600;color:{TEXT}">
      Generating Claude recommendation…</p>
    <p style="margin:0;font-size:0.75rem;color:{MUTED}">
      Synthesising LSTM + sentiment signals into portfolio advice</p>
  </div>
</div>""", unsafe_allow_html=True)

resp = requests.post("http://localhost:8000/synthesize", json={
    "amount": amount,   "currency": currency, "risk": risk,
    "horizon": horizon, "goal": goal,         "stop_loss": stop_loss,
    "existing": existing, "verbosity": verbosity,
    "lstm": lstm_r, "sentiment": sent_r, "news": news_r,
}).json()

prog_bar.empty()
status.empty()


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
alloc = resp["allocation"]
s     = resp["summary"]

# ── Summary banner ────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,{PRIMARY}18,{ACCENT}0a);
            border:1px solid {PRIMARY}44;border-radius:12px;
            padding:18px 24px;margin:20px 0 24px;
            display:flex;justify-content:space-between;
            align-items:center;flex-wrap:wrap;gap:12px">
  <div>
    <p style="margin:0;font-size:0.72rem;font-weight:700;color:{MUTED};
              text-transform:uppercase;letter-spacing:0.08em">Analysis Complete</p>
    <p style="margin:5px 0 0;font-size:0.9rem;color:{TEXT}">
      {len(sel_coins)} coins analysed &nbsp;·&nbsp;
      {len(alloc)} allocated &nbsp;·&nbsp;
      {len(sel_coins)-len(alloc)} skipped
    </p>
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap">
    <span style="background:{ACCENT}22;border:1px solid {ACCENT}44;border-radius:6px;
                 padding:4px 12px;font-size:0.78rem;color:{ACCENT};font-weight:700">
      ↑ {s['total_upside_pct']}% Upside</span>
    <span style="background:{DANGER}22;border:1px solid {DANGER}44;border-radius:6px;
                 padding:4px 12px;font-size:0.78rem;color:{DANGER};font-weight:700">
      ↓ {s['total_downside_pct']}% Downside</span>
    <span style="background:{risk_col}22;border:1px solid {risk_col}44;border-radius:6px;
                 padding:4px 12px;font-size:0.78rem;color:{risk_col};font-weight:700">
      {risk} Profile</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI metrics ───────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Upside",
          f"{s['total_upside_pct']}%",
          f"+{currency} {s['total_upside_inr']:,.0f}")
m2.metric("Total Downside",
          f"{s['total_downside_pct']}%",
          f"−{currency} {s['total_downside_inr']:,.0f}",
          delta_color="inverse")
m3.metric("Coins Allocated", len(alloc))
m4.metric("Coins Skipped",   len(sel_coins) - len(alloc))

st.markdown(f"<div style='height:24px'></div>", unsafe_allow_html=True)

# ── Row 1: Allocation donut + Upside/Downside bars ────────────────────────────
col1, col2 = st.columns([1, 1.1])

with col1:
    section_label("🥧  Portfolio Allocation")
    DONUT_COLORS = [PRIMARY, ACCENT, WARNING, PURPLE, DANGER,
                    CYAN, "#F97316", "#EC4899", "#14B8A6", "#6366F1"]
    fig1 = px.pie(
        names  = [f"{c} ({d['percent']}%)" for c, d in alloc.items()],
        values = [d["amount"] for d in alloc.values()],
        hole=0.5,
        color_discrete_sequence=DONUT_COLORS,
    )
    fig1.update_traces(
        textposition="inside",
        textinfo="label+percent",
        textfont=dict(color=TEXT, size=11, family="Inter"),
        hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<extra></extra>",
    )
    fig1.update_layout(
        **dark_chart_layout(height=340, showlegend=False),
        annotations=[dict(
            text=f"₹{amount:,.0f}", x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color=TEXT, family="Inter"),
        )],
    )
    st.plotly_chart(fig1, use_container_width=True,
                    config={"displayModeBar": False})

with col2:
    section_label("📊  Upside vs Downside by Coin")

    def get_upside(c):
        return lstm_r[c].get("upside", lstm_r[c].get("upside (%)", 0))

    def get_downside(c):
        return lstm_r[c].get("downside", lstm_r[c].get("downside (%)", 0))

    coins_l = list(lstm_r.keys())

    fig2 = go.Figure(data=[
        go.Bar(
            name="Upside %",
            x=coins_l,
            y=[get_upside(c) for c in coins_l],
            marker=dict(color=ACCENT, line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>Upside: %{y:.1f}%<extra></extra>",
        ),
        go.Bar(
            name="Downside %",
            x=coins_l,
            y=[-get_downside(c) for c in coins_l],
            marker=dict(color=DANGER, line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>Downside: %{y:.1f}%<extra></extra>",
        ),
    ])
    fig2.update_layout(
        **dark_chart_layout(height=340, barmode="overlay",
                            showlegend=True, bargap=0.25),
        yaxis=yax(title="Return (%)"),
        xaxis=xax(),
        legend=dict(
            font=dict(color=TEXT, size=11), bgcolor=SURFACE,
            bordercolor=BORDER, borderwidth=1,
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
    )
    st.plotly_chart(fig2, use_container_width=True,
                    config={"displayModeBar": False})

# ── Row 2: Sentiment bar chart ────────────────────────────────────────────────
st.markdown(f"<div style='height:8px'></div>", unsafe_allow_html=True)
section_label("🧠  FinBERT Sentiment — All Selected Coins")

sdf = pd.DataFrame([
    {"Coin": c, "Score": sv["score"], "Label": sv["label"]}
    for c, sv in sent_r.items()
]).sort_values("Score", ascending=False)

fig3 = px.bar(
    sdf, x="Coin", y="Score",
    color="Score",
    color_continuous_scale=[[0, DANGER], [0.5, WARNING], [1, ACCENT]],
    range_color=[-1, 1],
    text="Label",
)
fig3.update_traces(
    textposition="outside",
    textfont=dict(color=TEXT, size=11, family="Inter"),
    marker_line_width=0,
    hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>",
)
fig3.update_layout(
    **dark_chart_layout(height=320, showlegend=False),
    yaxis=yax(title="Sentiment Score", range=[-1.3, 1.3]),
    xaxis=xax(),
    coloraxis_colorbar=dict(
        tickfont=dict(color=MUTED),
        title=dict(text="Score", font=dict(color=MUTED)),
        bgcolor=SURFACE, bordercolor=BORDER,
    ),
)
fig3.add_hline(y=0, line_dash="dot", line_color=BORDER, line_width=1.5)
st.plotly_chart(fig3, use_container_width=True,
                config={"displayModeBar": False})

# ── Coin Detail Cards ─────────────────────────────────────────────────────────
st.markdown(f"<div style='height:8px'></div>", unsafe_allow_html=True)
section_label("🪙  Coin-Level Breakdown")

card_cols = st.columns(min(len(alloc), 4))
for idx, (coin, d) in enumerate(alloc.items()):
    sent  = sent_r.get(coin, {})
    lstm  = lstm_r.get(coin, {})
    s_col = ACCENT if sent.get("label") == "positive" else (
            DANGER if sent.get("label") == "negative" else MUTED)
    up    = get_upside(coin)
    dn    = get_downside(coin)

    card_cols[idx % 4].markdown(f"""
    <div style="background:{SURFACE};border:1px solid {BORDER};
                border-radius:12px;padding:16px 18px;margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;
                  align-items:center;margin-bottom:12px">
        <div>
          <p style="margin:0;font-size:1rem;font-weight:800;color:{TEXT}">{coin}</p>
          <p style="margin:2px 0 0;font-size:0.7rem;color:{MUTED}">
            {d['percent']}% allocation</p>
        </div>
        <div style="font-size:1.2rem;font-weight:900;color:{ACCENT}">
          ₹{d['amount']:,.0f}</div>
      </div>
      <div style="height:1px;background:{BORDER};margin-bottom:10px"></div>
      <div style="display:flex;justify-content:space-between;margin-bottom:6px">
        <span style="font-size:0.75rem;color:{MUTED}">Upside</span>
        <span style="font-size:0.75rem;font-weight:700;color:{ACCENT}">
          +{up:.1f}%</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:10px">
        <span style="font-size:0.75rem;color:{MUTED}">Downside</span>
        <span style="font-size:0.75rem;font-weight:700;color:{DANGER}">
          -{dn:.1f}%</span>
      </div>
      <span style="background:{s_col}22;color:{s_col};font-size:0.68rem;
                   font-weight:700;padding:3px 10px;border-radius:99px;
                   text-transform:uppercase;letter-spacing:0.04em">
        {sent.get('label','—')} sentiment</span>
    </div>
    """, unsafe_allow_html=True)

# ── Claude Recommendation ─────────────────────────────────────────────────────
st.markdown(f"<div style='height:8px'></div>", unsafe_allow_html=True)
section_label("🤖  Claude AI Recommendation")

st.markdown(f"""
<div style="background:{SURFACE};border:1px solid {PRIMARY}44;
            border-left:3px solid {PRIMARY};border-radius:12px;
            padding:22px 26px;line-height:1.8;font-size:0.9rem;color:{TEXT}">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
    <div style="width:28px;height:28px;background:{PRIMARY}22;border-radius:6px;
                display:flex;align-items:center;justify-content:center;
                font-size:0.9rem">🤖</div>
    <span style="font-weight:700;color:{PRIMARY};font-size:0.85rem;
                 letter-spacing:0.03em">ADVISOR OUTPUT</span>
    <span style="background:{SURFACE2};border:1px solid {BORDER};border-radius:4px;
                 padding:2px 8px;font-size:0.68rem;color:{MUTED}">{verbosity}</span>
  </div>
  {resp["advice"].replace(chr(10), "<br>")}
</div>
""", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="border-top:1px solid {BORDER};margin-top:32px;padding:14px 0;
            display:flex;justify-content:space-between;align-items:center;
            flex-wrap:wrap;gap:8px">
  <span style="font-size:0.73rem;color:{FAINT}">
    © 2024 CryptoAdvisor AI &nbsp;·&nbsp; LSTM + FinBERT + Claude
  </span>
  <span style="font-size:0.71rem;color:{FAINT}">
    ⚠️ Not financial advice · Do your own research
  </span>
</div>
""", unsafe_allow_html=True)
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="IDX Evidence Dashboard",
    page_icon="📈",
    layout="wide",
)

DATA = Path(__file__).parent / "data"

@st.cache_data
def load_data():
    price = pd.read_csv(DATA / "price_daily.csv", parse_dates=["Date"])
    own = pd.read_csv(DATA / "ownership_long.csv", parse_dates=["Date"])
    uni = pd.read_csv(DATA / "dashboard_universe.csv")
    for c in ["Open","High","Low","Close","AdjClose","Volume"]:
        price[c] = pd.to_numeric(price[c], errors="coerce")
    for c in ["RetailPct","DeltaRetail","SnapshotPrice","Coverage"]:
        own[c] = pd.to_numeric(own[c], errors="coerce")
    return price, own, uni

def rsi_wilder(s, n=14):
    delta=s.diff()
    gain=delta.clip(lower=0)
    loss=(-delta.clip(upper=0))
    avg_gain=gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss=loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs=avg_gain/avg_loss.replace(0,np.nan)
    rsi=100-(100/(1+rs))
    return rsi.fillna(100).where(avg_loss!=0,100)

def technical_frame(df):
    x=df.copy().sort_values("Date")
    x["SMA20"]=x["Close"].rolling(20).mean()
    x["SMA50"]=x["Close"].rolling(50).mean()
    x["SMA200"]=x["Close"].rolling(200).mean()
    x["RSI14"]=rsi_wilder(x["Close"],14)
    ema12=x["Close"].ewm(span=12, adjust=False).mean()
    ema26=x["Close"].ewm(span=26, adjust=False).mean()
    x["MACD"]=ema12-ema26
    x["MACDSignal"]=x["MACD"].ewm(span=9, adjust=False).mean()
    x["VolMA20"]=x["Volume"].rolling(20).mean()
    x["VolRatio"]=x["Volume"]/x["VolMA20"].replace(0,np.nan)
    return x

def technical_context(row):
    if pd.isna(row["SMA50"]):
        return "Insufficient data"
    c=row["Close"]
    if pd.notna(row["SMA200"]):
        if c>row["SMA20"]>row["SMA50"]>row["SMA200"]:
            return "Strong Uptrend"
        if c<row["SMA20"]<row["SMA50"]<row["SMA200"]:
            return "Strong Downtrend"
        if c>row["SMA50"] and row["SMA50"]>row["SMA200"]:
            return "Uptrend"
        if c<row["SMA50"] and row["SMA50"]<row["SMA200"]:
            return "Downtrend"
    return "Mixed / Sideways"

@st.cache_data
def build_breadth(price):
    universe = pd.read_csv(DATA / "dashboard_universe.csv")
    u100 = set(universe.loc[universe["Universe100"]=="Yes","Ticker"])
    p=price[price["Ticker"].isin(u100)].copy()
    frames=[]
    for t,g in p.groupby("Ticker"):
        g=technical_frame(g)
        g["Above20"]=g["Close"]>g["SMA20"]
        g["Above50"]=g["Close"]>g["SMA50"]
        g["Above200"]=g["Close"]>g["SMA200"]
        g["Ret20"]=g["Close"].pct_change(20)
        frames.append(g[["Date","Ticker","Above20","Above50","Above200","Ret20"]])
    z=pd.concat(frames,ignore_index=True)
    out=z.groupby("Date").agg(
        pct_above20=("Above20","mean"),
        pct_above50=("Above50","mean"),
        pct_above200=("Above200","mean"),
        median_ret20=("Ret20","median"),
    ).reset_index()
    return out

def regime_for_row(r, bull=0.55, bear=0.45):
    if pd.isna(r["median_ret20"]):
        return "Unknown"
    if r["pct_above20"]>=bull and r["pct_above50"]>=bull and r["median_ret20"]>0:
        return "Bull"
    if r["pct_above20"]<=bear and r["pct_above50"]<=bear and r["median_ret20"]<0:
        return "Bear"
    return "Neutral"

def regime_confidence(row):
    regs=[]
    for bull,bear in [(0.50,0.50),(0.55,0.45),(0.60,0.40)]:
        regs.append(regime_for_row(row,bull,bear))
    if len(set(regs))==1:
        return "High", regs
    if regs.count(regs[1])>=2:
        return "Medium", regs
    return "Low", regs

def latest_ownership_percentile(own, ticker, date):
    current = own[own["Date"]==date].copy()
    current = current[current["DeltaRetail"]<0].dropna(subset=["DeltaRetail"])
    if current.empty:
        return np.nan
    current["Magnitude"] = current["DeltaRetail"].abs()
    current["Percentile"] = current["Magnitude"].rank(method="average", pct=True)
    row=current[current["Ticker"]==ticker]
    return np.nan if row.empty else float(row["Percentile"].iloc[0])

def evidence_label(delta, pctile, volratio, regime, regime_conf, coverage):
    # Evidence system, deliberately not a buy/sell recommendation.
    score=0
    reasons=[]
    if pd.notna(delta):
        if delta<0:
            score += 1
            reasons.append("Retail ownership turun")
        elif delta>0:
            score -= 1
            reasons.append("Retail ownership naik")

    if pd.notna(pctile):
        if pctile>=0.90:
            score += 2
            reasons.append("penurunan Retail termasuk top 10%")
        elif pctile>=0.75:
            score += 1
            reasons.append("penurunan Retail termasuk top 25%")

    if pd.notna(volratio) and volratio>1:
        score += 1
        reasons.append("volume di atas MA20")

    if pd.notna(coverage):
        if coverage>=0.95:
            reasons.append("coverage KSEI tinggi")
        elif coverage<0.75:
            score -= 1
            reasons.append("coverage KSEI rendah")

    if regime=="Bear":
        score -= 2 if regime_conf=="High" else 1
        reasons.append("Bear regime risk override")
    elif regime=="Bull":
        score += 1
        reasons.append("market breadth Bull")

    if score>=4:
        label="Strong"
    elif score>=2:
        label="Moderate"
    elif score>=0:
        label="Weak"
    else:
        label="Caution"

    return label, score, reasons

price, own, uni = load_data()
breadth = build_breadth(price)

st.title("IDX Evidence Dashboard")
st.caption(
    "Ownership + Volume + Technical + Market Regime. "
    "Dashboard riset berbasis evidence, bukan rekomendasi beli/jual."
)

available = sorted(set(price["Ticker"]).intersection(set(own["Ticker"])))
default_idx = available.index("MEDC") if "MEDC" in available else 0

with st.sidebar:
    st.header("Filter")
    ticker = st.selectbox("Ticker", available, index=default_idx)
    lookback = st.selectbox("Chart period", ["6M","1Y","All"], index=1)
    st.divider()
    st.caption("Regime default: T55")
    st.caption("Bull ≥55% breadth; Bear ≤45% breadth.")

p = price[price["Ticker"]==ticker].copy().sort_values("Date")
# Exclude current/latest partially completed bar only if volume/price row date equals max overall and may be intraday.
# Data package was created after prior QC; latest complete technical view uses last row available.
tech = technical_frame(p)

if lookback=="6M":
    cutoff=tech["Date"].max()-pd.DateOffset(months=6)
    chart_df=tech[tech["Date"]>=cutoff]
elif lookback=="1Y":
    cutoff=tech["Date"].max()-pd.DateOffset(years=1)
    chart_df=tech[tech["Date"]>=cutoff]
else:
    chart_df=tech

latest=tech.iloc[-1]
own_t=own[(own["Ticker"]==ticker) & own["RetailPct"].notna()].sort_values("Date")
latest_own=own_t.iloc[-1] if not own_t.empty else None

# Market regime latest
b_latest=breadth.dropna(subset=["pct_above20","pct_above50"]).iloc[-1]
regime=regime_for_row(b_latest,0.55,0.45)
confidence, threshold_regs=regime_confidence(b_latest)

delta = latest_own["DeltaRetail"] if latest_own is not None else np.nan
coverage = latest_own["Coverage"] if latest_own is not None else np.nan
pctile = latest_ownership_percentile(own,ticker,latest_own["Date"]) if latest_own is not None else np.nan
volratio = latest["VolRatio"]

evidence, evidence_score, reasons = evidence_label(delta,pctile,volratio,regime,confidence,coverage)
trend=technical_context(latest)

# ---------- KPI row ----------
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Last Close", f"Rp {latest['Close']:,.0f}")
c2.metric("Retail %", "N/A" if latest_own is None else f"{latest_own['RetailPct']:.2%}",
          None if pd.isna(delta) else f"{delta:+.2%}")
c3.metric("Ownership Magnitude", "N/A" if pd.isna(pctile) else f"P{pctile*100:.0f}")
c4.metric("Volume Ratio", "N/A" if pd.isna(volratio) else f"{volratio:.2f}×")
c5.metric("Market Regime", regime, f"{confidence} confidence")
c6.metric("Evidence Level", evidence, f"internal evidence {evidence_score:+d}")

st.caption(
    f"Technical context: **{trend}** · RSI14: "
    f"{latest['RSI14']:.1f} · Coverage KSEI: "
    f"{'N/A' if pd.isna(coverage) else f'{coverage:.1%}'}"
)

# ---------- Main charts ----------
left,right = st.columns([1.6,1])

with left:
    st.subheader("Harga & Moving Average")
    fig=go.Figure()
    fig.add_trace(go.Candlestick(
        x=chart_df["Date"],
        open=chart_df["Open"],high=chart_df["High"],
        low=chart_df["Low"],close=chart_df["Close"],
        name="OHLC"
    ))
    fig.add_trace(go.Scatter(x=chart_df["Date"],y=chart_df["SMA20"],name="SMA20"))
    fig.add_trace(go.Scatter(x=chart_df["Date"],y=chart_df["SMA50"],name="SMA50"))
    fig.add_trace(go.Scatter(x=chart_df["Date"],y=chart_df["SMA200"],name="SMA200"))
    fig.update_layout(height=520,margin=dict(l=10,r=10,t=20,b=10),xaxis_rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True)

with right:
    st.subheader("Kepemilikan Retail KSEI")
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(
        x=own_t["Date"],y=own_t["RetailPct"]*100,
        mode="lines+markers",name="Retail %"
    ))
    fig2.update_layout(
        height=300,margin=dict(l=10,r=10,t=20,b=10),
        yaxis_title="Retail (%)"
    )
    st.plotly_chart(fig2,use_container_width=True)

    st.subheader("Perubahan Retail per Snapshot")
    bars=own_t.dropna(subset=["DeltaRetail"]).copy()
    fig3=go.Figure(go.Bar(x=bars["Date"],y=bars["DeltaRetail"]*100,name="Δ Retail"))
    fig3.update_layout(
        height=210,margin=dict(l=10,r=10,t=20,b=10),
        yaxis_title="Δ Retail (ppt)"
    )
    st.plotly_chart(fig3,use_container_width=True)

# ---------- Evidence panel ----------
st.subheader("Evidence Breakdown")
e1,e2,e3,e4 = st.columns(4)
e1.info(
    f"**Ownership**\n\n"
    f"{'Retail turun' if pd.notna(delta) and delta<0 else 'Retail naik / netral'}"
    f"\n\nMagnitude: {'N/A' if pd.isna(pctile) else f'P{pctile*100:.0f}'}"
)
e2.info(
    f"**Volume**\n\n"
    f"Volume Ratio: {'N/A' if pd.isna(volratio) else f'{volratio:.2f}× MA20'}"
)
e3.info(
    f"**Technical**\n\n{trend}\n\nRSI14: {latest['RSI14']:.1f}"
)
e4.info(
    f"**Market Regime**\n\n{regime}\n\nConfidence: {confidence}\n\n"
    f"T50/T55/T60: {' / '.join(threshold_regs)}"
)

with st.expander("Mengapa Evidence Level seperti ini?", expanded=True):
    if reasons:
        for r in reasons:
            st.write("•", r)
    else:
        st.write("Belum cukup evidence.")
    st.caption(
        "Evidence Level bukan probabilitas keuntungan dan bukan rekomendasi investasi. "
        "Bear regime diperlakukan sebagai risk override berdasarkan hasil Step 4F–4G."
    )

# ---------- Market breadth ----------
st.subheader("Market Breadth — Universe 100")
bchart=breadth.tail(180).copy()
fig4=go.Figure()
fig4.add_trace(go.Scatter(x=bchart["Date"],y=bchart["pct_above20"]*100,name="% > SMA20"))
fig4.add_trace(go.Scatter(x=bchart["Date"],y=bchart["pct_above50"]*100,name="% > SMA50"))
fig4.add_trace(go.Scatter(x=bchart["Date"],y=bchart["pct_above200"]*100,name="% > SMA200"))
fig4.add_hline(y=55,line_dash="dash",annotation_text="Bull threshold 55%")
fig4.add_hline(y=45,line_dash="dash",annotation_text="Bear threshold 45%")
fig4.update_layout(height=350,margin=dict(l=10,r=10,t=20,b=10),yaxis_title="% Universe")
st.plotly_chart(fig4,use_container_width=True)

# ---------- Detail table ----------
st.subheader("Snapshot Ownership")
detail=own_t[["Date","RetailPct","DeltaRetail","SnapshotPrice","Coverage"]].sort_values("Date",ascending=False).copy()
detail["RetailPct"]=detail["RetailPct"].map(lambda x: None if pd.isna(x) else f"{x:.2%}")
detail["DeltaRetail"]=detail["DeltaRetail"].map(lambda x: None if pd.isna(x) else f"{x:+.2%}")
detail["Coverage"]=detail["Coverage"].map(lambda x: None if pd.isna(x) else f"{x:.1%}")
detail["SnapshotPrice"]=detail["SnapshotPrice"].map(lambda x: None if pd.isna(x) else f"Rp {x:,.0f}")
st.dataframe(detail,use_container_width=True,hide_index=True)

st.divider()
st.caption(
    "Sumber data dashboard: snapshot kepemilikan KSEI yang diunggah pengguna; "
    "harga harian Yahoo Finance/yfinance sebagai ingestion sekunder yang telah diuji terhadap snapshot KSEI/IDX. "
    "Market regime dihitung dari Universe 100. Untuk riset/non-komersial."
)


st.divider()
latest_price_date = price["Date"].max()
latest_ownership_date = own["Date"].max()
f1, f2 = st.columns(2)
f1.metric("Harga terakhir dalam dataset", latest_price_date.strftime("%d %b %Y"))
f2.metric("Snapshot ownership terakhir", latest_ownership_date.strftime("%d %b %Y"))
st.caption("Step 5B: harga dapat diperbarui incremental; ownership hanya berubah saat snapshot KSEI baru tersedia.")

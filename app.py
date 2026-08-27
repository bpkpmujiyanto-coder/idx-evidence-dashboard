from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="IDX Evidence Dashboard",page_icon="📈",layout="wide")
DATA=Path(__file__).parent/"data"

@st.cache_data
def load_data():
    price=pd.read_csv(DATA/"price_daily.csv",parse_dates=["Date"])
    own=pd.read_csv(DATA/"ownership_long.csv",parse_dates=["Date"])
    uni=pd.read_csv(DATA/"dashboard_universe.csv")
    for c in ["Open","High","Low","Close","AdjClose","Volume"]:
        price[c]=pd.to_numeric(price[c],errors="coerce")
    for c in ["RetailPct","DeltaRetail","SnapshotPrice","Coverage"]:
        own[c]=pd.to_numeric(own[c],errors="coerce")
    price["Ticker"]=price["Ticker"].str.upper()
    own["Ticker"]=own["Ticker"].str.upper()
    uni["Ticker"]=uni["Ticker"].str.upper()
    return price,own,uni

def rsi(s,n=14):
    d=s.diff()
    g=d.clip(lower=0)
    l=(-d.clip(upper=0))
    ag=g.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    al=l.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    rs=ag/al.replace(0,np.nan)
    return 100-(100/(1+rs))

def technical_frame(df):
    x=df.sort_values("Date").copy()
    x["SMA20"]=x["Close"].rolling(20).mean()
    x["SMA50"]=x["Close"].rolling(50).mean()
    x["SMA200"]=x["Close"].rolling(200).mean()
    x["RSI14"]=rsi(x["Close"])
    e12=x["Close"].ewm(span=12,adjust=False).mean()
    e26=x["Close"].ewm(span=26,adjust=False).mean()
    x["MACD"]=e12-e26
    x["MACDSignal"]=x["MACD"].ewm(span=9,adjust=False).mean()
    x["VolMA20"]=x["Volume"].rolling(20).mean()
    x["VolRatio"]=x["Volume"]/x["VolMA20"].replace(0,np.nan)
    return x

def trend_label(r):
    if pd.isna(r["SMA50"]): return "Insufficient"
    c=r["Close"]
    if pd.notna(r["SMA200"]):
        if c>r["SMA20"]>r["SMA50"]>r["SMA200"]: return "Strong Uptrend"
        if c<r["SMA20"]<r["SMA50"]<r["SMA200"]: return "Strong Downtrend"
        if c>r["SMA50"] and r["SMA50"]>r["SMA200"]: return "Uptrend"
        if c<r["SMA50"] and r["SMA50"]<r["SMA200"]: return "Downtrend"
    return "Mixed / Above SMA50" if c>r["SMA50"] else "Mixed / Below SMA50"

@st.cache_data
def build_breadth(price,uni):
    u100=set(uni.loc[uni["Universe100"]=="Yes","Ticker"])
    frames=[]
    for t,g in price[price["Ticker"].isin(u100)].groupby("Ticker"):
        z=technical_frame(g)
        z["A20"]=z["Close"]>z["SMA20"]
        z["A50"]=z["Close"]>z["SMA50"]
        z["A200"]=z["Close"]>z["SMA200"]
        z["R20"]=z["Close"].pct_change(20)
        frames.append(z[["Date","A20","A50","A200","R20"]])
    x=pd.concat(frames)
    return x.groupby("Date").agg(
        pct_above20=("A20","mean"),pct_above50=("A50","mean"),
        pct_above200=("A200","mean"),median_ret20=("R20","median")
    ).reset_index()

def regime(r,bull=.55,bear=.45):
    if pd.isna(r["median_ret20"]): return "Unknown"
    if r["pct_above20"]>=bull and r["pct_above50"]>=bull and r["median_ret20"]>0: return "Bull"
    if r["pct_above20"]<=bear and r["pct_above50"]<=bear and r["median_ret20"]<0: return "Bear"
    return "Neutral"

def regime_confidence(r):
    regs=[regime(r,.50,.50),regime(r,.55,.45),regime(r,.60,.40)]
    return ("High" if len(set(regs))==1 else "Medium"),regs

def add_percentile(own):
    x=own.copy()
    x["Magnitude"]=x["DeltaRetail"].abs()
    x["OwnershipPctile"]=np.nan
    for d,g in x.groupby("Date"):
        idx=g.index[(g["DeltaRetail"]<0)&g["DeltaRetail"].notna()]
        if len(idx):
            x.loc[idx,"OwnershipPctile"]=x.loc[idx,"Magnitude"].rank(pct=True)
    return x

def evidence(delta,pctile,vr,market_regime,conf,coverage,trend):
    score=0; pos=[]; risk=[]
    if pd.notna(delta):
        if delta<0: score+=1; pos.append("Retail ownership turun")
        elif delta>0: score-=1; risk.append("Retail ownership naik")
    if pd.notna(pctile):
        if pctile>=.90: score+=2; pos.append("penurunan Retail top 10%")
        elif pctile>=.75: score+=1; pos.append("penurunan Retail top 25%")
    if pd.notna(vr) and vr>1: score+=1; pos.append("volume > MA20")
    if pd.notna(coverage):
        if coverage>=.95: pos.append("coverage KSEI tinggi")
        elif coverage<.75: score-=1; risk.append("coverage KSEI rendah")
    if trend in ("Uptrend","Strong Uptrend"): score+=1; pos.append("technical trend positif")
    elif trend in ("Downtrend","Strong Downtrend"): score-=1; risk.append("technical trend lemah")
    if market_regime=="Bear":
        score-=2 if conf=="High" else 1
        risk.append("Bear regime risk override")
    elif market_regime=="Bull":
        score+=1; pos.append("market breadth Bull")
    label="Strong" if score>=4 else "Moderate" if score>=2 else "Weak" if score>=0 else "Caution"
    return label,score,pos,risk

price,own,uni=load_data()
own=add_percentile(own)
breadth=build_breadth(price,uni)
bnow=breadth.dropna(subset=["pct_above20","pct_above50"]).iloc[-1]
mreg=regime(bnow)
rconf,rtests=regime_confidence(bnow)

@st.cache_data
def make_screener(price,own,uni,mreg,rconf):
    rows=[]
    u100=set(uni.loc[uni["Universe100"]=="Yes","Ticker"])
    for t in sorted(set(price["Ticker"])&set(own["Ticker"])):
        p=price[price["Ticker"]==t]
        if len(p)<20: continue
        tf=technical_frame(p)
        last=tf.iloc[-1]
        ot=own[(own["Ticker"]==t)&own["RetailPct"].notna()].sort_values("Date")
        if ot.empty: continue
        lo=ot.iloc[-1]
        tr=trend_label(last)
        lab,score,pos,risk=evidence(lo["DeltaRetail"],lo["OwnershipPctile"],last["VolRatio"],mreg,rconf,lo["Coverage"],tr)
        rows.append({
            "Ticker":t,"Close":last["Close"],"RetailPct":lo["RetailPct"],
            "DeltaRetail":lo["DeltaRetail"],"OwnershipPctile":lo["OwnershipPctile"],
            "Coverage":lo["Coverage"],"VolRatio":last["VolRatio"],"RSI14":last["RSI14"],
            "Trend":tr,"Evidence":lab,"EvidenceScore":score,
            "RiskOverride":"Yes" if any("Bear regime" in x for x in risk) else "No",
            "Universe100":"Yes" if t in u100 else "No"
        })
    return pd.DataFrame(rows)

scr=make_screener(price,own,uni,mreg,rconf)

st.title("IDX Evidence Dashboard")
st.caption("Ownership + Volume + Technical + Market Regime. Untuk riset, bukan rekomendasi beli/jual.")

m1,m2,m3,m4=st.columns(4)
m1.metric("Market Regime",mreg,f"{rconf} confidence")
m2.metric("% > SMA20",f"{bnow['pct_above20']:.1%}")
m3.metric("% > SMA50",f"{bnow['pct_above50']:.1%}")
m4.metric("Median Return 20D",f"{bnow['median_ret20']:.1%}")

tab1,tab2,tab3=st.tabs(["🔎 Screener Universe","📊 Detail Ticker","🌐 Market Regime"])

with tab1:
    st.subheader("Universe Screener")
    st.caption("Urutan berdasarkan evidence strength, bukan expected return.")
    c1,c2,c3,c4,c5=st.columns(5)
    ev=c1.multiselect("Evidence",["Strong","Moderate","Weak","Caution"],default=["Strong","Moderate"])
    univ=c2.selectbox("Universe",["All","Universe 100","Extended"])
    ownf=c3.selectbox("Ownership",["All","Top 10% decline","Top 25% decline","Retail decline","Retail increase"])
    trends=c4.multiselect("Trend",sorted(scr["Trend"].dropna().unique()))
    riskf=c5.selectbox("Risk override",["All","No","Yes"])
    mincov=st.slider("Minimum KSEI coverage",0,100,75,5)/100

    v=scr.copy()
    if ev: v=v[v["Evidence"].isin(ev)]
    if univ=="Universe 100": v=v[v["Universe100"]=="Yes"]
    elif univ=="Extended": v=v[v["Universe100"]=="No"]
    if ownf=="Top 10% decline": v=v[(v["DeltaRetail"]<0)&(v["OwnershipPctile"]>=.90)]
    elif ownf=="Top 25% decline": v=v[(v["DeltaRetail"]<0)&(v["OwnershipPctile"]>=.75)]
    elif ownf=="Retail decline": v=v[v["DeltaRetail"]<0]
    elif ownf=="Retail increase": v=v[v["DeltaRetail"]>0]
    if trends: v=v[v["Trend"].isin(trends)]
    if riskf!="All": v=v[v["RiskOverride"]==riskf]
    v=v[(v["Coverage"].isna())|(v["Coverage"]>=mincov)]
    v=v.sort_values(["EvidenceScore","OwnershipPctile","VolRatio"],ascending=[False,False,False])

    a,b,c,d=st.columns(4)
    a.metric("Candidates",len(v))
    b.metric("Strong",int((v["Evidence"]=="Strong").sum()))
    c.metric("Top 10% ownership",int((v["OwnershipPctile"]>=.90).sum()))
    d.metric("Volume > MA20",int((v["VolRatio"]>1).sum()))

    disp=v[["Ticker","Evidence","EvidenceScore","RiskOverride","Close","RetailPct","DeltaRetail","OwnershipPctile","VolRatio","RSI14","Trend","Coverage"]].copy()
    disp["Close"]=disp["Close"].map(lambda x:f"Rp {x:,.0f}")
    disp["RetailPct"]=disp["RetailPct"].map(lambda x:"" if pd.isna(x) else f"{x:.2%}")
    disp["DeltaRetail"]=disp["DeltaRetail"].map(lambda x:"" if pd.isna(x) else f"{x:+.2%}")
    disp["OwnershipPctile"]=disp["OwnershipPctile"].map(lambda x:"" if pd.isna(x) else f"P{x*100:.0f}")
    disp["VolRatio"]=disp["VolRatio"].map(lambda x:"" if pd.isna(x) else f"{x:.2f}×")
    disp["RSI14"]=disp["RSI14"].map(lambda x:"" if pd.isna(x) else f"{x:.1f}")
    disp["Coverage"]=disp["Coverage"].map(lambda x:"" if pd.isna(x) else f"{x:.1%}")
    st.dataframe(disp,use_container_width=True,hide_index=True,height=520)

    st.subheader("Most Interesting to Research")
    for _,r in v.head(10).iterrows():
        st.markdown(
            f"**{r['Ticker']} — {r['Evidence']}** · "
            f"Ownership {'P'+str(int(r['OwnershipPctile']*100)) if pd.notna(r['OwnershipPctile']) else 'N/A'} · "
            f"Vol {r['VolRatio']:.2f}× · {r['Trend']}"
        )

with tab2:
    ticker=st.selectbox("Ticker",sorted(set(price["Ticker"])&set(own["Ticker"])),index=0)
    p=price[price["Ticker"]==ticker]
    tf=technical_frame(p)
    ot=own[(own["Ticker"]==ticker)&own["RetailPct"].notna()].sort_values("Date")
    last=tf.iloc[-1]; lo=ot.iloc[-1]; tr=trend_label(last)
    lab,score,pos,risk=evidence(lo["DeltaRetail"],lo["OwnershipPctile"],last["VolRatio"],mreg,rconf,lo["Coverage"],tr)

    k1,k2,k3,k4,k5,k6=st.columns(6)
    k1.metric("Close",f"Rp {last['Close']:,.0f}")
    k2.metric("Retail %",f"{lo['RetailPct']:.2%}",f"{lo['DeltaRetail']:+.2%}" if pd.notna(lo["DeltaRetail"]) else None)
    k3.metric("Ownership","N/A" if pd.isna(lo["OwnershipPctile"]) else f"P{lo['OwnershipPctile']*100:.0f}")
    k4.metric("Volume Ratio","N/A" if pd.isna(last["VolRatio"]) else f"{last['VolRatio']:.2f}×")
    k5.metric("Trend",tr)
    k6.metric("Evidence",lab,f"score {score:+d}")

    left,right=st.columns([1.65,1])
    with left:
        fig=go.Figure()
        fig.add_trace(go.Candlestick(x=tf["Date"],open=tf["Open"],high=tf["High"],low=tf["Low"],close=tf["Close"],name="OHLC"))
        fig.add_trace(go.Scatter(x=tf["Date"],y=tf["SMA20"],name="SMA20"))
        fig.add_trace(go.Scatter(x=tf["Date"],y=tf["SMA50"],name="SMA50"))
        fig.add_trace(go.Scatter(x=tf["Date"],y=tf["SMA200"],name="SMA200"))
        fig.update_layout(height=520,xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        fig2=go.Figure(go.Scatter(x=ot["Date"],y=ot["RetailPct"]*100,mode="lines+markers",name="Retail %"))
        fig2.update_layout(height=280,margin=dict(l=10,r=10,t=20,b=10),yaxis_title="Retail (%)")
        st.plotly_chart(fig2,use_container_width=True)
        fig3=go.Figure(go.Bar(x=ot["Date"],y=ot["DeltaRetail"]*100,name="Δ Retail"))
        fig3.update_layout(height=230,margin=dict(l=10,r=10,t=20,b=10),yaxis_title="Δ Retail (ppt)")
        st.plotly_chart(fig3,use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        st.markdown("#### Positive evidence")
        for x in pos: st.write("•",x)
        if not pos: st.write("Belum ada positive evidence kuat.")
    with c2:
        st.markdown("#### Risk / invalidation context")
        for x in risk: st.write("•",x)
        if not risk: st.write("Belum ada risk override utama.")

with tab3:
    st.subheader("Market Breadth")
    st.write(f"Current regime: **{mreg}** · confidence **{rconf}** · T50/T55/T60: {' / '.join(rtests)}")
    b=breadth.tail(220)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=b["Date"],y=b["pct_above20"]*100,name="% > SMA20"))
    fig.add_trace(go.Scatter(x=b["Date"],y=b["pct_above50"]*100,name="% > SMA50"))
    fig.add_trace(go.Scatter(x=b["Date"],y=b["pct_above200"]*100,name="% > SMA200"))
    fig.add_hline(y=55,line_dash="dash",annotation_text="Bull 55%")
    fig.add_hline(y=45,line_dash="dash",annotation_text="Bear 45%")
    fig.update_layout(height=430,margin=dict(l=10,r=10,t=20,b=10),yaxis_title="% Universe")
    st.plotly_chart(fig,use_container_width=True)

st.divider()
f1,f2=st.columns(2)
f1.metric("Harga terakhir",price["Date"].max().strftime("%d %b %Y"))
f2.metric("Ownership snapshot",own["Date"].max().strftime("%d %b %Y"))
st.caption("Evidence Level bukan probabilitas keuntungan. Ownership tidak identik dengan satu pelaku pasar.")

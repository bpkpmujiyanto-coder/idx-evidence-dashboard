from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="IDX Evidence Dashboard",page_icon="📈",layout="wide")
DATA=Path(__file__).parent/"data"

# ---------------- DATA ----------------
@st.cache_data
def load_data():
    price=pd.read_csv(DATA/"price_daily.csv",parse_dates=["Date"])
    own=pd.read_csv(DATA/"ownership_long.csv",parse_dates=["Date"])
    uni=pd.read_csv(DATA/"dashboard_universe.csv")

    for c in ["Open","High","Low","Close","AdjClose","Volume"]:
        if c in price.columns:
            price[c]=pd.to_numeric(price[c],errors="coerce")
    for c in ["RetailPct","DeltaRetail","SnapshotPrice","Coverage"]:
        own[c]=pd.to_numeric(own[c],errors="coerce")

    price["Ticker"]=price["Ticker"].astype(str).str.upper()
    own["Ticker"]=own["Ticker"].astype(str).str.upper()
    uni["Ticker"]=uni["Ticker"].astype(str).str.upper()
    return price,own,uni

def completed_eod_only(df):
    if df.empty:
        return df
    now=pd.Timestamp.now(tz=ZoneInfo("Asia/Jakarta"))
    today=now.date()
    # Before ~16:20 WIB, exclude today's potentially unfinished daily candle.
    if now.hour<16 or (now.hour==16 and now.minute<20):
        return df[df["Date"].dt.date<today].copy()
    return df.copy()

@st.cache_data(ttl=3600,show_spinner=False)
def fetch_ticker_on_demand(ticker,start="2025-03-27"):
    yt=f"{ticker}.JK"
    try:
        x=yf.download(
            yt,start=start,interval="1d",auto_adjust=False,actions=False,
            progress=False,multi_level_index=False,threads=False
        )
        if x is None or x.empty:
            return pd.DataFrame()
        x=x.reset_index().rename(columns={"Adj Close":"AdjClose"})
        x["Ticker"]=ticker
        wanted=["Date","Ticker","Open","High","Low","Close","AdjClose","Volume"]
        for c in wanted:
            if c not in x.columns: x[c]=pd.NA
        x=x[wanted]
        x["Date"]=pd.to_datetime(x["Date"])
        for c in ["Open","High","Low","Close","AdjClose","Volume"]:
            x[c]=pd.to_numeric(x[c],errors="coerce")
        return completed_eod_only(x)
    except Exception:
        return pd.DataFrame()

def get_price_history(price,ticker,min_rows=120):
    local=price[price["Ticker"]==ticker].copy().sort_values("Date")
    local=completed_eod_only(local)
    if len(local)>=min_rows:
        return local,"Histori lokal lengkap"
    remote=fetch_ticker_on_demand(ticker)
    if not remote.empty and len(remote)>len(local):
        return remote,"Histori diambil saat dibuka"
    if not local.empty:
        return local,"Histori lokal masih pendek"
    return pd.DataFrame(),"Harga belum tersedia"

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
    x["RVOL20"]=x["VolRatio"]
    x["Ret5D"]=x["Close"].pct_change(5)
    x["Ret20D"]=x["Close"].pct_change(20)
    x["Ret60D"]=x["Close"].pct_change(60)
    x["DistSMA20"]=x["Close"]/x["SMA20"]-1
    x["DistSMA50"]=x["Close"]/x["SMA50"]-1
    x["High20"]=x["High"].rolling(20).max()
    x["Low20"]=x["Low"].rolling(20).min()
    x["High60"]=x["High"].rolling(60).max()
    x["Low60"]=x["Low"].rolling(60).min()
    x["PrevHigh20"]=x["High"].shift(1).rolling(20).max()
    x["PrevLow20"]=x["Low"].shift(1).rolling(20).min()
    x["BreakoutDistance"]=x["Close"]/x["PrevHigh20"]-1

    prev=x["Close"].shift(1)
    tr=pd.concat([
        x["High"]-x["Low"],
        (x["High"]-prev).abs(),
        (x["Low"]-prev).abs()
    ],axis=1).max(axis=1)
    x["ATR14"]=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()

    up_move=x["High"].diff()
    down_move=-x["Low"].diff()
    plus_dm=pd.Series(np.where((up_move>down_move)&(up_move>0),up_move,0.0),index=x.index)
    minus_dm=pd.Series(np.where((down_move>up_move)&(down_move>0),down_move,0.0),index=x.index)
    atr_w=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    plus_sm=plus_dm.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    minus_sm=minus_dm.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    x["PlusDI14"]=100*plus_sm/atr_w.replace(0,np.nan)
    x["MinusDI14"]=100*minus_sm/atr_w.replace(0,np.nan)
    dx=100*(x["PlusDI14"]-x["MinusDI14"]).abs()/(x["PlusDI14"]+x["MinusDI14"]).replace(0,np.nan)
    x["ADX14"]=dx.ewm(alpha=1/14,adjust=False,min_periods=14).mean()

    h=x["High"]; l=x["Low"]
    x["PivotHigh"]=np.where(
        (h.shift(2)>h.shift(4))&(h.shift(2)>h.shift(3))&
        (h.shift(2)>h.shift(1))&(h.shift(2)>h),
        h.shift(2),np.nan
    )
    x["PivotLow"]=np.where(
        (l.shift(2)<l.shift(4))&(l.shift(2)<l.shift(3))&
        (l.shift(2)<l.shift(1))&(l.shift(2)<l),
        l.shift(2),np.nan
    )
    return x

def trend_label(r):
    if pd.isna(r.get("SMA50")): return "Data belum cukup"
    c=r["Close"]
    if pd.notna(r.get("SMA200")):
        if c>r["SMA20"]>r["SMA50"]>r["SMA200"]: return "Naik kuat"
        if c<r["SMA20"]<r["SMA50"]<r["SMA200"]: return "Turun kuat"
        if c>r["SMA50"] and r["SMA50"]>r["SMA200"]: return "Cenderung naik"
        if c<r["SMA50"] and r["SMA50"]<r["SMA200"]: return "Cenderung turun"
    return "Netral / sideways"

@st.cache_data
def build_breadth(price,uni):
    u100=set(uni.loc[uni["Universe100"]=="Yes","Ticker"])
    frames=[]
    for _,g in price[price["Ticker"].isin(u100)].groupby("Ticker"):
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

def regime_raw(r,bull=.55,bear=.45):
    if pd.isna(r["median_ret20"]): return "Unknown"
    if r["pct_above20"]>=bull and r["pct_above50"]>=bull and r["median_ret20"]>0: return "Bull"
    if r["pct_above20"]<=bear and r["pct_above50"]<=bear and r["median_ret20"]<0: return "Bear"
    return "Neutral"

def regime_text(raw):
    return {"Bull":"Pasar sehat","Bear":"Pasar lemah","Neutral":"Pasar netral"}.get(raw,"Belum diketahui")

def regime_confidence(r):
    regs=[regime_raw(r,.50,.50),regime_raw(r,.55,.45),regime_raw(r,.60,.40)]
    return ("Tinggi" if len(set(regs))==1 else "Sedang"),regs

def add_percentile(own):
    x=own.copy()
    x["Magnitude"]=x["DeltaRetail"].abs()
    x["OwnershipPctile"]=np.nan
    for _,g in x.groupby("Date"):
        idx=g.index[(g["DeltaRetail"]<0)&g["DeltaRetail"].notna()]
        if len(idx):
            x.loc[idx,"OwnershipPctile"]=x.loc[idx,"Magnitude"].rank(pct=True)
    return x

def ownership_text(delta,p):
    if pd.isna(delta): return "Belum ada perubahan"
    if delta>0: return "Porsi ritel meningkat"
    if pd.notna(p) and p>=.90: return "Ritel turun sangat besar"
    if pd.notna(p) and p>=.75: return "Ritel turun cukup besar"
    return "Ritel sedikit berkurang"

def volume_text(v):
    if pd.isna(v): return "Data volume belum cukup"
    if v>=1.5: return "Volume sangat ramai"
    if v>=1: return "Volume di atas normal"
    if v>=.75: return "Volume normal"
    return "Volume sepi"

def momentum_text(v):
    if pd.isna(v): return "Belum tersedia"
    if v>=70: return "Sudah cukup panas"
    if v>=55: return "Momentum positif"
    if v>=45: return "Momentum netral"
    if v>=30: return "Momentum lemah"
    return "Tekanan jual tinggi"

def evidence(delta,pctile,vr,mreg_raw,conf,coverage,trend,has_ownership=True):
    score=0; pos=[]; risk=[]
    if has_ownership and pd.notna(delta):
        if delta<0: score+=1; pos.append("Porsi kepemilikan ritel berkurang")
        elif delta>0: score-=1; risk.append("Porsi kepemilikan ritel bertambah")
    if has_ownership and pd.notna(pctile):
        if pctile>=.90: score+=2; pos.append("Penurunan ritel termasuk 10% terbesar")
        elif pctile>=.75: score+=1; pos.append("Penurunan ritel termasuk 25% terbesar")
    if pd.notna(vr) and vr>1: score+=1; pos.append("Aktivitas transaksi lebih ramai dari biasanya")
    if has_ownership and pd.notna(coverage):
        if coverage>=.95: pos.append("Kualitas data ownership tinggi")
        elif coverage<.75: score-=1; risk.append("Coverage data ownership rendah")
    if trend in ("Cenderung naik","Naik kuat"): score+=1; pos.append("Arah harga sedang mendukung")
    elif trend in ("Cenderung turun","Turun kuat"): score-=1; risk.append("Arah harga masih lemah")
    if mreg_raw=="Bear":
        score-=2 if conf=="Tinggi" else 1; risk.append("Kondisi pasar sedang lemah")
    elif mreg_raw=="Bull":
        score+=1; pos.append("Kondisi pasar sedang sehat")

    if score>=4: label="Menarik"
    elif score>=2: label="Perlu dipantau"
    elif score>=0: label="Netral"
    else: label="Waspada"
    return label,score,pos,risk

def action_bucket(label,risk_override,trend):
    if label=="Menarik" and risk_override=="Tidak" and trend not in ("Turun kuat","Cenderung turun"):
        return "Kandidat Akumulasi"
    if label in ("Menarik","Perlu dipantau"): return "Watchlist"
    return "Hindari dulu"

def idx_tick_size(p):
    if p<200:return 1
    if p<500:return 2
    if p<2000:return 5
    if p<5000:return 10
    return 25

def round_tick(v,mode="nearest"):
    if pd.isna(v):return np.nan
    tick=idx_tick_size(v)
    if mode=="down":return np.floor(v/tick)*tick
    if mode=="up":return np.ceil(v/tick)*tick
    return np.round(v/tick)*tick

def rupiah(v):
    return "N/A" if pd.isna(v) else f"Rp {v:,.0f}"

def trading_plan(tf,status,trend,mreg):
    last=tf.iloc[-1]
    close=float(last["Close"])
    atr=float(last["ATR14"]) if pd.notna(last["ATR14"]) else close*.03
    adx=float(last["ADX14"]) if pd.notna(last["ADX14"]) else np.nan
    plusdi=float(last["PlusDI14"]) if pd.notna(last["PlusDI14"]) else np.nan
    minusdi=float(last["MinusDI14"]) if pd.notna(last["MinusDI14"]) else np.nan
    rvol=float(last["RVOL20"]) if pd.notna(last["RVOL20"]) else np.nan

    piv_hi=tf.loc[tf["PivotHigh"].notna(),"PivotHigh"]
    piv_lo=tf.loc[tf["PivotLow"].notna(),"PivotLow"]
    last_piv_hi=float(piv_hi.iloc[-1]) if len(piv_hi) else np.nan
    last_piv_lo=float(piv_lo.iloc[-1]) if len(piv_lo) else np.nan

    support_candidates=[last.get("SMA20",np.nan),last.get("SMA50",np.nan),last.get("PrevLow20",np.nan),last_piv_lo]
    support_candidates=[float(v) for v in support_candidates if pd.notna(v) and v<=close]
    support=max(support_candidates) if support_candidates else close-atr

    resistance_candidates=[last.get("PrevHigh20",np.nan),last_piv_hi]
    resistance_candidates=[float(v) for v in resistance_candidates if pd.notna(v) and v>=close*.98]
    resistance=min(resistance_candidates) if resistance_candidates else close+1.5*atr

    invalidation=support-.75*atr

    aggressive_low=max(support-.15*atr,0)
    aggressive_high=support+.45*atr
    aggressive_trigger="Harga mendekati support tanpa breakdown."
    aggressive_ok=(trend not in ("Turun kuat","Cenderung turun") and close<=support+1.25*atr and mreg!="Pasar lemah")

    cons_ref=max([v for v in [last.get("SMA20",np.nan),last.get("SMA50",np.nan)] if pd.notna(v)] or [close])
    conservative_low=max(cons_ref,close-.25*atr)
    conservative_high=close+.40*atr
    conservative_trigger="Harga bertahan di atas rata-rata 20/50 hari dan arah tren terkonfirmasi."
    conservative_ok=(
        pd.notna(adx) and adx>=20 and
        pd.notna(plusdi) and pd.notna(minusdi) and plusdi>minusdi and
        pd.notna(last.get("SMA20")) and close>=last["SMA20"] and
        trend in ("Naik kuat","Cenderung naik","Netral / sideways") and
        mreg!="Pasar lemah"
    )

    breakout_trigger=round_tick(resistance+idx_tick_size(resistance),"up")
    breakout_low=breakout_trigger
    breakout_high=round_tick(breakout_trigger+.50*atr,"up")
    breakout_trigger_text="Harga menembus resistance sebelumnya dengan volume relatif kuat."
    breakout_now=(close>=resistance and pd.notna(rvol) and rvol>=1.5 and pd.notna(plusdi) and pd.notna(minusdi) and plusdi>minusdi)
    breakout_ready=(pd.notna(rvol) and rvol>=1.0 and pd.notna(adx) and adx>=18 and pd.notna(plusdi) and pd.notna(minusdi) and plusdi>minusdi)

    target1=max(resistance+1.0*atr,close+1.25*atr)
    target2=max(target1+1.5*atr,close+3.0*atr)

    def rr(entry):
        risk=max(entry-invalidation,.0001)
        return (target1-entry)/risk,(target2-entry)/risk

    ag_mid=(aggressive_low+aggressive_high)/2
    co_mid=(conservative_low+conservative_high)/2
    br_mid=(breakout_low+breakout_high)/2
    ag_rr1,ag_rr2=rr(ag_mid)
    co_rr1,co_rr2=rr(co_mid)
    br_rr1,br_rr2=rr(br_mid)

    if mreg=="Pasar lemah":
        recommended="Tunggu"
        reason="Pasar sedang lemah; jangan memaksakan entry hanya karena level teknikal terlihat menarik."
    elif breakout_now:
        recommended="Breakout Entry"
        reason="Resistance sudah ditembus dan volume relatif kuat. Tetap tunggu harga tidak kembali jatuh ke bawah resistance."
    elif conservative_ok:
        recommended="Entry Konservatif"
        reason="Arah harga dan DMI sudah mendukung, dengan ADX yang menunjukkan tren mulai cukup kuat."
    elif aggressive_ok:
        recommended="Entry Agresif"
        reason="Harga masih dekat area support. Skenario ini lebih dini sehingga risiko false signal lebih tinggi."
    elif breakout_ready:
        recommended="Tunggu Breakout"
        reason="Momentum dan arah mulai mendukung, tetapi harga belum menembus resistance secara valid."
    else:
        recommended="Tunggu"
        reason="Belum ada kombinasi support, tren, dan volume yang cukup kuat."

    return dict(
        support=round_tick(support),resistance=round_tick(resistance),
        invalidation=round_tick(invalidation,"down"),
        target1=round_tick(target1,"up"),target2=round_tick(target2,"up"),
        adx=adx,plusdi=plusdi,minusdi=minusdi,rvol=rvol,
        recommended=recommended,reason=reason,
        aggressive_low=round_tick(aggressive_low,"down"),aggressive_high=round_tick(aggressive_high,"up"),
        aggressive_ok=aggressive_ok,aggressive_trigger=aggressive_trigger,
        aggressive_rr1=ag_rr1,aggressive_rr2=ag_rr2,
        conservative_low=round_tick(conservative_low,"down"),conservative_high=round_tick(conservative_high,"up"),
        conservative_ok=conservative_ok,conservative_trigger=conservative_trigger,
        conservative_rr1=co_rr1,conservative_rr2=co_rr2,
        breakout_low=round_tick(breakout_low,"up"),breakout_high=round_tick(breakout_high,"up"),
        breakout_now=breakout_now,breakout_ready=breakout_ready,
        breakout_trigger=breakout_trigger_text,breakout_rr1=br_rr1,breakout_rr2=br_rr2
    )

def analyst_card(row,mreg):
    pos=[]; risk=[]; waits=[]; invalid=[]
    if row.get("HasOwnership",False):
        if row["DeltaRetail"]<0:
            if pd.notna(row["OwnershipPctile"]) and row["OwnershipPctile"]>=.90: pos.append("Penurunan kepemilikan ritel termasuk sangat besar.")
            elif pd.notna(row["OwnershipPctile"]) and row["OwnershipPctile"]>=.75: pos.append("Kepemilikan ritel turun cukup besar.")
            else: pos.append("Kepemilikan ritel berkurang.")
        elif row["DeltaRetail"]>0: risk.append("Porsi kepemilikan ritel meningkat.")
    else:
        waits.append("Data ownership belum tersedia untuk ticker ini.")

    if pd.notna(row["VolRatio"]) and row["VolRatio"]>1: pos.append("Aktivitas transaksi lebih ramai dari rata-rata.")
    else: waits.append("Tunggu aktivitas transaksi meningkat.")

    if row["Trend"] in ("Naik kuat","Cenderung naik"): pos.append("Arah harga sedang mendukung.")
    elif row["Trend"] in ("Turun kuat","Cenderung turun"): risk.append("Arah harga masih lemah.")
    else: waits.append("Tunggu arah harga menjadi lebih jelas.")

    if mreg=="Pasar lemah": risk.append("Kondisi pasar sedang lemah.")
    elif mreg=="Pasar netral": waits.append("Pasar belum benar-benar kuat.")

    invalid.append("Tesis teknikal melemah jika harga kembali konsisten di bawah rata-rata 50 hari.")
    if row.get("HasOwnership",False) and row["DeltaRetail"]<0:
        invalid.append("Tesis ownership melemah jika snapshot berikutnya menunjukkan ritel kembali meningkat tajam.")
    invalid.append("Keyakinan harus diturunkan jika market regime berubah menjadi Pasar lemah.")
    return pos or ["Belum ada faktor positif dominan."],risk or ["Belum ada risk override utama dari model."],waits or ["Pantau harga, volume, dan data berikutnya."],invalid


def technical_glossary():
    return {
        "ADX": "ADX mengukur seberapa kuat sebuah tren. ADX tidak menunjukkan arah. Semakin tinggi, tren biasanya semakin kuat.",
        "DMI": "DMI membantu membaca arah tren. +DI mewakili tekanan naik, sedangkan -DI mewakili tekanan turun.",
        "+DI/-DI": "Jika +DI berada di atas -DI, tekanan naik lebih dominan. Jika -DI lebih tinggi, tekanan turun lebih dominan.",
        "RVOL": "Relative Volume membandingkan volume terbaru dengan rata-rata volume 20 hari. Contoh 1,5x berarti transaksi 50% lebih ramai dari biasanya.",
        "Support": "Area harga yang sebelumnya cenderung menahan penurunan. Support bukan jaminan harga akan memantul.",
        "Resistance": "Area harga yang sebelumnya cenderung menahan kenaikan. Jika ditembus dengan kuat, resistance dapat menjadi sinyal breakout.",
        "Breakout": "Breakout terjadi saat harga menembus resistance. Breakout lebih meyakinkan jika volume ikut meningkat.",
        "ATR": "ATR mengukur besar kecilnya pergerakan harga atau volatilitas. Dashboard memakainya untuk memberi jarak yang wajar pada batas batal dan target.",
        "RR": "Risk–Reward membandingkan potensi keuntungan dengan risiko. RR 2x berarti potensi target kira-kira dua kali risiko menuju batas batal.",
        "Invalidation": "Batas batal adalah level yang membuat skenario teknikal perlu dievaluasi ulang jika ditembus.",
        "SMA20/50/200": "Rata-rata harga 20, 50, dan 200 hari. Dipakai untuk melihat arah harga jangka pendek, menengah, dan panjang."
    }

def fundamental_label(value, good_condition, caution_condition=None):
    if pd.isna(value):
        return "Data belum tersedia"
    if good_condition(value):
        return "Baik"
    if caution_condition is not None and caution_condition(value):
        return "Perlu perhatian"
    return "Netral"

def fmt_percent(v):
    return "N/A" if pd.isna(v) else f"{v:.1%}"

def fmt_number(v, suffix=""):
    if pd.isna(v):
        return "N/A"
    if abs(v)>=1e12:
        return f"{v/1e12:.1f} T{suffix}"
    if abs(v)>=1e9:
        return f"{v/1e9:.1f} M{suffix}"
    if abs(v)>=1e6:
        return f"{v/1e6:.1f} Jt{suffix}"
    return f"{v:,.0f}{suffix}"

@st.cache_data(ttl=21600,show_spinner=False)
def fetch_fundamental_snapshot(ticker):
    """
    Secondary market-data source. Values may be incomplete and should be verified
    against the latest IDX/company filing before high-stakes use.
    """
    try:
        info=yf.Ticker(f"{ticker}.JK").info or {}
    except Exception:
        return {}

    def g(key):
        v=info.get(key,np.nan)
        try:
            return float(v) if v is not None else np.nan
        except Exception:
            return np.nan

    return {
        "company": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector") or "N/A",
        "industry": info.get("industry") or "N/A",
        "marketCap": g("marketCap"),
        "trailingPE": g("trailingPE"),
        "forwardPE": g("forwardPE"),
        "priceToBook": g("priceToBook"),
        "returnOnEquity": g("returnOnEquity"),
        "returnOnAssets": g("returnOnAssets"),
        "debtToEquity": g("debtToEquity"),
        "revenueGrowth": g("revenueGrowth"),
        "earningsGrowth": g("earningsGrowth"),
        "grossMargins": g("grossMargins"),
        "operatingMargins": g("operatingMargins"),
        "profitMargins": g("profitMargins"),
        "dividendYield": g("dividendYield"),
        "operatingCashflow": g("operatingCashflow"),
        "freeCashflow": g("freeCashflow"),
        "totalCash": g("totalCash"),
        "totalDebt": g("totalDebt"),
        "currentRatio": g("currentRatio"),
        "quickRatio": g("quickRatio"),
        "enterpriseToEbitda": g("enterpriseToEbitda"),
        "enterpriseToRevenue": g("enterpriseToRevenue"),
    }

def fundamental_interpretation(f):
    if not f:
        return "Data fundamental belum berhasil diperoleh.", [], []

    positives=[]
    risks=[]

    rg=f.get("revenueGrowth",np.nan)
    eg=f.get("earningsGrowth",np.nan)
    roe=f.get("returnOnEquity",np.nan)
    pm=f.get("profitMargins",np.nan)
    de=f.get("debtToEquity",np.nan)
    ocf=f.get("operatingCashflow",np.nan)
    fcf=f.get("freeCashflow",np.nan)
    pe=f.get("trailingPE",np.nan)
    pb=f.get("priceToBook",np.nan)

    if pd.notna(rg):
        if rg>0.10: positives.append("Pendapatan tumbuh cukup kuat.")
        elif rg<0: risks.append("Pendapatan sedang menurun.")
    if pd.notna(eg):
        if eg>0.10: positives.append("Pertumbuhan laba cukup kuat.")
        elif eg<0: risks.append("Laba sedang menurun.")
    if pd.notna(roe):
        if roe>=0.15: positives.append("ROE menunjukkan kemampuan menghasilkan laba atas modal yang baik.")
        elif roe<0.05: risks.append("ROE masih rendah.")
    if pd.notna(pm):
        if pm>0.10: positives.append("Margin laba bersih relatif sehat.")
        elif pm<0: risks.append("Margin laba bersih negatif.")
    if pd.notna(de):
        # Yahoo typically reports debtToEquity in percent-like units (e.g., 50 = 0.5x).
        der=de/100
        if der>2: risks.append("Utang relatif tinggi terhadap ekuitas.")
        elif der<1: positives.append("Leverage relatif terkendali.")
    if pd.notna(ocf):
        if ocf>0: positives.append("Arus kas operasi positif.")
        else: risks.append("Arus kas operasi negatif.")
    if pd.notna(fcf):
        if fcf<0: risks.append("Free cash flow negatif.")
    if pd.notna(pe) and pe<=0:
        risks.append("PER tidak bermakna/negatif karena laba tidak positif.")
    if pd.notna(pb) and pb>5:
        risks.append("PBV terlihat tinggi; perlu dibandingkan dengan sektor dan historis.")

    if positives and len(positives)>=len(risks)+2:
        summary="Fundamental terlihat cukup sehat dari data sekunder yang tersedia."
    elif risks and len(risks)>len(positives):
        summary="Ada beberapa area fundamental yang perlu diperiksa lebih lanjut."
    else:
        summary="Fundamental terlihat campuran; belum cukup kuat untuk simpulan tunggal."

    return summary,positives,risks

price,own,uni=load_data()
own=add_percentile(own)
breadth=build_breadth(price,uni)
bnow=breadth.dropna(subset=["pct_above20","pct_above50"]).iloc[-1]
mreg_raw=regime_raw(bnow); mreg=regime_text(mreg_raw); rconf,rtests=regime_confidence(bnow)

@st.cache_data
def make_screener(price,own,uni,mreg_raw,rconf):
    rows=[]
    u100=set(uni.loc[uni["Universe100"]=="Yes","Ticker"])
    for t,g in price.groupby("Ticker"):
        if len(g)<60: continue
        tf=technical_frame(g)
        last=tf.iloc[-1]; tr=trend_label(last)
        ot=own[(own["Ticker"]==t)&own["RetailPct"].notna()].sort_values("Date")
        hasown=not ot.empty
        if hasown:
            lo=ot.iloc[-1]; delta=lo["DeltaRetail"]; pctile=lo["OwnershipPctile"]; cov=lo["Coverage"]; retail=lo["RetailPct"]
        else:
            delta=pctile=cov=retail=np.nan
        lab,score,pos,risk=evidence(delta,pctile,last["VolRatio"],mreg_raw,rconf,cov,tr,hasown)
        override="Ya" if any("pasar" in x.lower() and "lemah" in x.lower() for x in risk) else "Tidak"
        status=action_bucket(lab,override,tr)
        rows.append(dict(
            Ticker=t,Status=status,Kesimpulan=lab,EvidenceScore=score,Close=last["Close"],
            Ret20D=last["Ret20D"],RetailPct=retail,DeltaRetail=delta,OwnershipPctile=pctile,
            OwnershipText=ownership_text(delta,pctile) if hasown else "Ownership belum tersedia",
            Coverage=cov,VolRatio=last["VolRatio"],VolumeText=volume_text(last["VolRatio"]),
            RSI14=last["RSI14"],MomentumText=momentum_text(last["RSI14"]),Trend=tr,
            RiskOverride=override,Universe100="Ya" if t in u100 else "Tidak",HasOwnership=hasown
        ))
    return pd.DataFrame(rows)

scr=make_screener(price,own,uni,mreg_raw,rconf)
full_universe=sorted(uni["Ticker"].dropna().unique())
local_price_count=price["Ticker"].nunique()

# ---------------- UI ----------------
st.title("IDX Evidence Dashboard")
st.caption("Full IDX Universe — semua ticker pada snapshot KSEI terbaru dapat dicari. Bahasa utama dibuat sederhana untuk pengguna awam.")

c1,c2,c3,c4=st.columns(4)
c1.metric("Kondisi pasar",mreg,f"Keyakinan {rconf.lower()}")
c2.metric("Saham di atas MA20",f"{bnow['pct_above20']:.0%}")
c3.metric("Saham di atas MA50",f"{bnow['pct_above50']:.0%}")
c4.metric("Cakupan ticker",f"{len(full_universe):,} saham")

if mreg=="Pasar sehat": st.success("🟢 Pasar cukup mendukung. Sinyal individual lebih layak diperhatikan.")
elif mreg=="Pasar lemah": st.error("🔴 Pasar sedang lemah. Sinyal individual perlu diperlakukan lebih hati-hati.")
else: st.info("🟡 Pasar sedang netral. Pilih saham secara selektif dan tunggu konfirmasi.")

if local_price_count<len(full_universe):
    st.info(
        f"ℹ️ **Cakupan Full IDX:** {len(full_universe):,} ticker sudah bisa dicari di Detail Saham. "
        f"Screener saat ini memiliki histori lokal untuk {local_price_count:,} ticker. "
        "Ticker yang belum punya histori lokal akan mengambil harga saat dibuka; jalankan workflow **Bootstrap Full IDX Prices** "
        "untuk melengkapi seluruh screener."
    )

tab1,tab2,tab3,tab4,tab5=st.tabs(["⭐ Pilihan Hari Ini","🔎 Semua Saham","📊 Detail Saham","🏢 Fundamental","🌐 Kondisi Pasar"])

with tab1:
    st.subheader("Pilihan Hari Ini")
    st.write("Saham di bawah adalah kandidat **untuk diteliti lebih lanjut**, bukan daftar rekomendasi beli.")
    focus=scr.sort_values(["EvidenceScore","OwnershipPctile","VolRatio"],ascending=[False,False,False])
    a,b,c=st.columns(3)
    a.metric("Kandidat Akumulasi",int((focus["Status"]=="Kandidat Akumulasi").sum()))
    b.metric("Watchlist",int((focus["Status"]=="Watchlist").sum()))
    c.metric("Hindari dulu",int((focus["Status"]=="Hindari dulu").sum()))
    for _,r in focus[focus["Status"].isin(["Kandidat Akumulasi","Watchlist"])].head(12).iterrows():
        with st.container(border=True):
            x1,x2,x3,x4=st.columns([1,1.1,1.5,3])
            x1.markdown(f"### {'🟢' if r['Status']=='Kandidat Akumulasi' else '🟡'} {r['Ticker']}")
            x1.write(f"**{r['Status']}**")
            x2.metric("Harga",rupiah(r["Close"]))
            x2.caption(f"20 hari: {r['Ret20D']:+.1%}" if pd.notna(r["Ret20D"]) else "20 hari: N/A")
            x3.write(f"**Ownership**\n\n{r['OwnershipText']}")
            x3.write(f"**Volume**\n\n{r['VolumeText']}")
            x4.write(f"Arah harga: **{r['Trend']}**. Momentum: **{r['MomentumText']}**. Pasar: **{mreg}**.")

with tab2:
    st.subheader("Semua Saham — Screener")
    st.caption("Screener hanya menghitung ticker dengan histori harga lokal yang cukup. Semua ticker tetap tersedia di Detail Saham.")
    f1,f2,f3=st.columns(3)
    sf=f1.multiselect("Status",["Kandidat Akumulasi","Watchlist","Hindari dulu"],default=["Kandidat Akumulasi","Watchlist"])
    of=f2.selectbox("Ownership",["Semua","Ritel turun sangat besar","Ritel turun cukup besar","Ritel sedikit berkurang","Porsi ritel meningkat","Ownership belum tersedia"])
    tfilt=f3.selectbox("Arah harga",["Semua","Naik kuat","Cenderung naik","Netral / sideways","Cenderung turun","Turun kuat"])
    v=scr.copy()
    if sf:v=v[v["Status"].isin(sf)]
    if of!="Semua":v=v[v["OwnershipText"]==of]
    if tfilt!="Semua":v=v[v["Trend"]==tfilt]
    v=v.sort_values(["EvidenceScore","OwnershipPctile","VolRatio"],ascending=[False,False,False])
    disp=v[["Ticker","Status","Close","Ret20D","OwnershipText","VolumeText","Trend","MomentumText"]].copy()
    disp.columns=["Kode","Status","Harga","20 Hari","Ownership","Volume","Arah Harga","Momentum"]
    disp["Harga"]=disp["Harga"].map(rupiah)
    disp["20 Hari"]=disp["20 Hari"].map(lambda x:"" if pd.isna(x) else f"{x:+.1%}")
    st.dataframe(disp,use_container_width=True,hide_index=True,height=560)

with tab3:
    st.subheader("Detail Saham — Full IDX")
    ticker=st.selectbox("Pilih / ketik kode saham",full_universe,index=full_universe.index("ERAA") if "ERAA" in full_universe else 0)
    with st.spinner(f"Menyiapkan data {ticker}..."):
        p,price_status=get_price_history(price,ticker)

    ot=own[(own["Ticker"]==ticker)&own["RetailPct"].notna()].sort_values("Date")
    hasown=not ot.empty
    if p.empty:
        st.error(f"Data harga {ticker} belum berhasil diperoleh. Ticker tetap tercantum karena ada di master KSEI EQUITY.")
        if hasown:
            lo=ot.iloc[-1]
            st.info(f"Ownership tersedia: Retail {lo['RetailPct']:.2%}, coverage {lo['Coverage']:.1%}.")
    else:
        tf=technical_frame(p)
        last=tf.iloc[-1]; tr=trend_label(last)
        if hasown:
            lo=ot.iloc[-1]; delta=lo["DeltaRetail"]; pctile=lo["OwnershipPctile"]; cov=lo["Coverage"]; retail=lo["RetailPct"]
        else:
            delta=pctile=cov=retail=np.nan

        lab,score,pos,risk=evidence(delta,pctile,last["VolRatio"],mreg_raw,rconf,cov,tr,hasown)
        override="Ya" if any("pasar" in x.lower() and "lemah" in x.lower() for x in risk) else "Tidak"
        status=action_bucket(lab,override,tr)

        row=dict(
            Ticker=ticker,Status=status,Close=last["Close"],Ret20D=last["Ret20D"],
            RetailPct=retail,DeltaRetail=delta,OwnershipPctile=pctile,Coverage=cov,
            VolRatio=last["VolRatio"],RSI14=last["RSI14"],Trend=tr,HasOwnership=hasown
        )

        st.caption(f"Status data harga: **{price_status}** · Ownership: **{'Tersedia' if hasown else 'Belum tersedia'}**")
        if status=="Kandidat Akumulasi": st.success(f"**{ticker}: Kandidat Akumulasi** — beberapa bukti utama muncul bersamaan.")
        elif status=="Watchlist": st.warning(f"**{ticker}: Watchlist** — ada tanda menarik, tetapi konfirmasi belum lengkap.")
        else: st.info(f"**{ticker}: Hindari dulu / Tunggu** — faktor pendukung belum cukup kuat.")

        m1,m2,m3,m4,m5=st.columns(5)
        m1.metric("Harga",rupiah(last["Close"]))
        m2.metric("20 hari","N/A" if pd.isna(last["Ret20D"]) else f"{last['Ret20D']:+.1%}")
        m3.metric("Ownership",ownership_text(delta,pctile) if hasown else "Belum tersedia")
        m4.metric("Volume",volume_text(last["VolRatio"]))
        m5.metric("Arah harga",tr)

        st.markdown("### Cara membacanya")
        q1,q2,q3,q4=st.columns(4)
        q1.info(f"**1. Ownership**\n\n{ownership_text(delta,pctile) if hasown else 'Data ownership belum tersedia'}")
        q2.info(f"**2. Aktivitas transaksi**\n\n{volume_text(last['VolRatio'])}")
        q3.info(f"**3. Arah harga**\n\n{tr}")
        q4.info(f"**4. Momentum**\n\n{momentum_text(last['RSI14'])}")

        st.markdown("### Kartu Analisis Saham")
        positives,risks,waits,invalid=analyst_card(row,mreg)
        a1,a2=st.columns(2)
        with a1:
            with st.container(border=True):
                st.markdown("#### ✅ Apa yang menarik?")
                for x in positives:st.write("•",x)
            with st.container(border=True):
                st.markdown("#### ⏳ Apa yang perlu ditunggu?")
                for x in waits:st.write("•",x)
        with a2:
            with st.container(border=True):
                st.markdown("#### ⚠️ Apa risikonya?")
                for x in risks:st.write("•",x)
            with st.container(border=True):
                st.markdown("#### ❌ Kapan tesis dianggap batal?")
                for x in invalid:st.write("•",x)

        if len(tf)>=60:
            st.markdown("### Advanced Entry Plan")
            plan=trading_plan(tf,status,tr,mreg)

            if plan["recommended"] in ("Breakout Entry","Entry Konservatif"):
                st.success(f"**Skenario yang paling sesuai saat ini: {plan['recommended']}** — {plan['reason']}")
            elif plan["recommended"]=="Entry Agresif":
                st.warning(f"**Skenario yang paling sesuai saat ini: {plan['recommended']}** — {plan['reason']}")
            else:
                st.info(f"**Skenario saat ini: {plan['recommended']}** — {plan['reason']}")

            adx_txt="N/A" if pd.isna(plan["adx"]) else f"{plan['adx']:.1f}"
            rvol_txt="N/A" if pd.isna(plan["rvol"]) else f"{plan['rvol']:.2f}x"
            di_txt="N/A" if pd.isna(plan["plusdi"]) or pd.isna(plan["minusdi"]) else f"{plan['plusdi']:.1f}/{plan['minusdi']:.1f}"
            st.caption(f"ADX14: {adx_txt} · RVOL20: {rvol_txt} · +DI/-DI: {di_txt}")
            st.info(
                "**Cara membaca indikator ini:** "
                "ADX menunjukkan **kekuatan tren**; +DI dibanding -DI menunjukkan **arah tekanan harga**; "
                "RVOL menunjukkan apakah transaksi **lebih ramai atau lebih sepi dari biasanya**."
            )

            st.markdown("#### Pilihan skenario entry")
            ec1,ec2,ec3=st.columns(3)

            with ec1:
                with st.container(border=True):
                    st.markdown("##### 🟠 Entry Agresif")
                    st.write(f"**Area:** {rupiah(plan['aggressive_low'])} – {rupiah(plan['aggressive_high'])}")
                    st.write("Masuk lebih awal dekat support.")
                    st.caption(plan["aggressive_trigger"])
                    st.write(f"RR Target 1: **{plan['aggressive_rr1']:.1f}x**")
                    st.write("✅ Relevan sekarang" if plan["aggressive_ok"] else "⏳ Belum ideal")

            with ec2:
                with st.container(border=True):
                    st.markdown("##### 🟢 Entry Konservatif")
                    st.write(f"**Area:** {rupiah(plan['conservative_low'])} – {rupiah(plan['conservative_high'])}")
                    st.write("Menunggu tren lebih terkonfirmasi.")
                    st.caption(plan["conservative_trigger"])
                    st.write(f"RR Target 1: **{plan['conservative_rr1']:.1f}x**")
                    st.write("✅ Relevan sekarang" if plan["conservative_ok"] else "⏳ Belum terkonfirmasi")

            with ec3:
                with st.container(border=True):
                    st.markdown("##### 🔵 Breakout Entry")
                    st.write(f"**Area:** {rupiah(plan['breakout_low'])} – {rupiah(plan['breakout_high'])}")
                    st.write(f"Resistance: **{rupiah(plan['resistance'])}**")
                    st.caption(plan["breakout_trigger"])
                    st.write(f"RR Target 1: **{plan['breakout_rr1']:.1f}x**")
                    if plan["breakout_now"]:
                        st.write("✅ Breakout sedang aktif")
                    elif plan["breakout_ready"]:
                        st.write("👀 Siap dipantau untuk breakout")
                    else:
                        st.write("⏳ Belum siap breakout")

            st.markdown("#### Level risiko dan target")
            t1,t2,t3,t4=st.columns(4)
            t1.metric("Support utama",rupiah(plan["support"]))
            t2.metric("Batas batal",rupiah(plan["invalidation"]))
            t3.metric("Target 1",rupiah(plan["target1"]))
            t4.metric("Target 2",rupiah(plan["target2"]))

            with st.expander("📘 Penjelasan istilah teknis"):
                glossary=technical_glossary()
                for term,desc in glossary.items():
                    st.write(f"**{term}** — {desc}")

            with st.expander("Cara membaca tiga jenis entry"):
                st.write("**Entry Agresif**: masuk lebih dini dekat support. Potensi harga masuk lebih murah, tetapi risiko salah timing lebih besar.")
                st.write("**Entry Konservatif**: menunggu tren dan arah DMI lebih jelas. Harga masuk bisa lebih tinggi, tetapi probabilitas false start diharapkan lebih rendah.")
                st.write("**Breakout Entry**: menunggu resistance ditembus dan idealnya RVOL meningkat. Cocok untuk saham yang sedang keluar dari area konsolidasi.")
                st.write("**ADX** mengukur kekuatan tren, bukan arah. +DI di atas -DI menunjukkan arah positif lebih dominan.")
                st.write("**RVOL20** membandingkan volume terbaru terhadap rata-rata 20 hari. Nilai >1,5x dianggap konfirmasi volume yang kuat dalam model ini.")
        else:
            plan=None
            st.warning("Histori harga belum cukup panjang untuk membuat Advanced Entry Plan.")

        left,right=st.columns([1.65,1])
        with left:
            st.markdown("#### Grafik harga")
            fig=go.Figure()
            fig.add_trace(go.Candlestick(x=tf["Date"],open=tf["Open"],high=tf["High"],low=tf["Low"],close=tf["Close"],name="Harga"))
            fig.add_trace(go.Scatter(x=tf["Date"],y=tf["SMA20"],name="Rata-rata 20 hari"))
            fig.add_trace(go.Scatter(x=tf["Date"],y=tf["SMA50"],name="Rata-rata 50 hari"))
            fig.add_trace(go.Scatter(x=tf["Date"],y=tf["SMA200"],name="Rata-rata 200 hari"))
            if plan:
                fig.add_hline(y=plan["support"],line_dash="dot",annotation_text="Support")
                fig.add_hline(y=plan["resistance"],line_dash="dot",annotation_text="Resistance")
                fig.add_hline(y=plan["invalidation"],line_dash="dash",annotation_text="Batas batal")
                fig.add_hline(y=plan["target1"],line_dash="dot",annotation_text="Target 1")
                fig.add_hline(y=plan["target2"],line_dash="dot",annotation_text="Target 2")
            fig.update_layout(height=500,xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig,use_container_width=True)
        with right:
            st.markdown("#### Kepemilikan ritel")
            if hasown:
                fig2=go.Figure(go.Scatter(x=ot["Date"],y=ot["RetailPct"]*100,mode="lines+markers",name="Retail"))
                fig2.update_layout(height=300,margin=dict(l=10,r=10,t=20,b=10),yaxis_title="% Retail")
                st.plotly_chart(fig2,use_container_width=True)
            else:
                st.info("Belum ada data ownership untuk ticker ini.")

with tab4:
    st.subheader("Fundamental Snapshot")
    st.write(
        "Bagian ini menjawab pertanyaan sederhana: **bisnisnya tumbuh, menguntungkan, dan sehat atau tidak?** "
        "Data otomatis di bawah berasal dari Yahoo Finance sebagai **sumber sekunder**. "
        "Untuk keputusan penting, verifikasi dengan laporan keuangan/keterbukaan informasi IDX terbaru."
    )

    fticker=st.selectbox(
        "Pilih saham untuk fundamental",
        full_universe,
        index=full_universe.index("ERAA") if "ERAA" in full_universe else 0,
        key="fundamental_ticker"
    )

    with st.spinner(f"Mengambil fundamental {fticker}..."):
        f=fetch_fundamental_snapshot(fticker)

    if not f:
        st.warning("Data fundamental otomatis belum tersedia untuk ticker ini.")
    else:
        st.caption(f"Perusahaan: **{f['company']}** · Sektor: **{f['sector']}** · Industri: **{f['industry']}**")

        fsum,fpos,frisk=fundamental_interpretation(f)
        if frisk and len(frisk)>len(fpos):
            st.warning(f"**Ringkasan:** {fsum}")
        else:
            st.info(f"**Ringkasan:** {fsum}")

        st.markdown("### 1. Pertumbuhan & Profitabilitas")
        g1,g2,g3,g4=st.columns(4)
        g1.metric("Pertumbuhan pendapatan",fmt_percent(f["revenueGrowth"]))
        g1.caption("Apakah penjualan bertambah dibanding periode sebelumnya.")
        g2.metric("Pertumbuhan laba",fmt_percent(f["earningsGrowth"]))
        g2.caption("Apakah laba berkembang, bukan hanya penjualan.")
        g3.metric("ROE",fmt_percent(f["returnOnEquity"]))
        g3.caption("Kemampuan perusahaan menghasilkan laba dari modal pemegang saham.")
        g4.metric("Margin laba bersih",fmt_percent(f["profitMargins"]))
        g4.caption("Berapa bagian penjualan yang akhirnya menjadi laba bersih.")

        st.markdown("### 2. Utang & Arus Kas")
        h1,h2,h3,h4=st.columns(4)
        der=np.nan if pd.isna(f["debtToEquity"]) else f["debtToEquity"]/100
        h1.metric("DER / Utang : Ekuitas","N/A" if pd.isna(der) else f"{der:.2f}x")
        h1.caption("Semakin tinggi, semakin besar ketergantungan pada utang. Tidak cocok dibandingkan lintas sektor tanpa konteks.")
        h2.metric("Kas",fmt_number(f["totalCash"]))
        h2.caption("Kas dan setara kas yang dilaporkan oleh sumber sekunder.")
        h3.metric("Arus kas operasi",fmt_number(f["operatingCashflow"]))
        h3.caption("Kas yang dihasilkan oleh operasi utama perusahaan.")
        h4.metric("Free cash flow",fmt_number(f["freeCashflow"]))
        h4.caption("Kas setelah kebutuhan investasi utama; negatif tidak selalu buruk jika perusahaan sedang ekspansi.")

        st.markdown("### 3. Valuasi")
        v1,v2,v3,v4=st.columns(4)
        v1.metric("PER","N/A" if pd.isna(f["trailingPE"]) else f"{f['trailingPE']:.1f}x")
        v1.caption("Harga saham dibanding laba. PER rendah belum otomatis murah.")
        v2.metric("PBV","N/A" if pd.isna(f["priceToBook"]) else f"{f['priceToBook']:.1f}x")
        v2.caption("Harga dibanding nilai buku. Sangat tergantung karakter sektor.")
        v3.metric("EV/EBITDA","N/A" if pd.isna(f["enterpriseToEbitda"]) else f"{f['enterpriseToEbitda']:.1f}x")
        v3.caption("Valuasi perusahaan terhadap EBITDA; lebih relevan untuk bisnis non-keuangan.")
        v4.metric("Dividend yield",fmt_percent(f["dividendYield"]))
        v4.caption("Perkiraan imbal hasil dividen berdasarkan data sekunder.")

        st.markdown("### Apa yang terlihat baik?")
        if fpos:
            for x in fpos:
                st.write("✅",x)
        else:
            st.write("Belum ada sinyal fundamental positif yang cukup jelas dari data otomatis.")

        st.markdown("### Apa yang perlu diperiksa?")
        if frisk:
            for x in frisk:
                st.write("⚠️",x)
        else:
            st.write("Belum ada peringatan utama dari data otomatis.")

        with st.expander("📘 Penjelasan istilah fundamental"):
            st.write("**PER** — berapa kali harga saham dibanding laba per saham. Harus dibandingkan dengan historis dan peer.")
            st.write("**PBV** — harga dibanding nilai buku perusahaan. Sangat relevan untuk sektor tertentu seperti bank, tetapi interpretasinya berbeda di sektor lain.")
            st.write("**ROE** — kemampuan menghasilkan laba dari modal pemegang saham.")
            st.write("**DER** — perbandingan utang dengan ekuitas. DER tinggi dapat meningkatkan risiko saat laba melemah atau bunga naik.")
            st.write("**Margin laba** — bagian pendapatan yang tersisa menjadi laba.")
            st.write("**Arus kas operasi** — kas nyata dari kegiatan utama. Laba yang bagus tetapi arus kas operasi buruk perlu diperiksa.")
            st.write("**Free cash flow** — kas operasi setelah belanja investasi utama.")
            st.write("**Dividend yield** — dividen relatif terhadap harga saham.")
            st.write("**EV/EBITDA** — valuasi perusahaan termasuk utang dibanding EBITDA. Tidak cocok untuk bank/lembaga keuangan.")

        if "Financial" in str(f["sector"]) or "Bank" in str(f["industry"]):
            st.warning(
                "Untuk bank/lembaga keuangan, metrik seperti DER dan EV/EBITDA kurang tepat. "
                "Tahap berikutnya akan menggunakan NIM, CASA, LDR, CAR, NPL, CoC, dan pertumbuhan kredit jika sumber datanya tersedia."
            )

with tab5:
    st.subheader("Kondisi Pasar")
    st.write("Market regime tetap dihitung dari Universe 100 agar indikator pasar tidak terlalu dipengaruhi saham sangat kecil/illiquid.")
    b=breadth.tail(220)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=b["Date"],y=b["pct_above20"]*100,name="% saham di atas MA20"))
    fig.add_trace(go.Scatter(x=b["Date"],y=b["pct_above50"]*100,name="% saham di atas MA50"))
    fig.add_hline(y=55,line_dash="dash",annotation_text="Pasar sehat")
    fig.add_hline(y=45,line_dash="dash",annotation_text="Pasar lemah")
    fig.update_layout(height=430,margin=dict(l=10,r=10,t=20,b=10),yaxis_title="% saham")
    st.plotly_chart(fig,use_container_width=True)

st.divider()
d1,d2,d3=st.columns(3)
d1.metric("Master saham",f"{len(full_universe):,}")
d2.metric("Histori harga lokal",f"{local_price_count:,}")
d3.metric("Snapshot ownership terakhir",own["Date"].max().strftime("%d %b %Y"))
st.caption(
    "Master Full IDX pada versi ini berasal dari saham bertipe EQUITY di snapshot KSEI terbaru yang tersedia dalam data proyek. "
    "Ticker baru setelah snapshot tersebut perlu ditambahkan saat snapshot KSEI berikutnya masuk. "
    "Label dashboard adalah alat bantu riset, bukan rekomendasi investasi."
)

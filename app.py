from pathlib import Path
from zoneinfo import ZoneInfo
import urllib.parse
import re
import urllib.request
import xml.etree.ElementTree as ET
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

def _statement_row(df, candidates):
    if df is None or getattr(df,"empty",True):
        return None
    for name in candidates:
        if name in df.index:
            s=pd.to_numeric(df.loc[name],errors="coerce").dropna()
            if not s.empty:
                return s
    # fuzzy fallback
    idx_lower={str(i).lower():i for i in df.index}
    for cand in candidates:
        c=cand.lower()
        for low,orig in idx_lower.items():
            if c in low or low in c:
                s=pd.to_numeric(df.loc[orig],errors="coerce").dropna()
                if not s.empty:
                    return s
    return None

def _latest_value(series):
    if series is None or len(series)==0:
        return np.nan
    return float(series.iloc[0])

def _sum_latest(series,n=4):
    if series is None or len(series)<n:
        return np.nan
    return float(series.iloc[:n].sum())

def _yoy_from_quarters(series):
    if series is None or len(series)<8:
        return np.nan
    cur=float(series.iloc[:4].sum())
    prev=float(series.iloc[4:8].sum())
    if prev==0:
        return np.nan
    return cur/prev-1

def _annual_growth(series):
    if series is None or len(series)<2:
        return np.nan
    cur=float(series.iloc[0]); prev=float(series.iloc[1])
    if prev==0:
        return np.nan
    return cur/prev-1

def _safe_div(a,b):
    if pd.isna(a) or pd.isna(b) or b==0:
        return np.nan
    return a/b

def _get_market_price_and_cap(obj,ticker):
    price=np.nan; market_cap=np.nan; shares=np.nan
    try:
        fi=obj.fast_info
        price=float(getattr(fi,"last_price",np.nan) or np.nan)
        market_cap=float(getattr(fi,"market_cap",np.nan) or np.nan)
    except Exception:
        pass
    if pd.isna(price):
        try:
            h=obj.history(period="5d",auto_adjust=False)
            if not h.empty:
                price=float(h["Close"].dropna().iloc[-1])
        except Exception:
            pass
    if pd.isna(market_cap):
        try:
            info=obj.info or {}
            market_cap=float(info.get("marketCap") or np.nan)
            shares=float(info.get("sharesOutstanding") or np.nan)
            if pd.isna(market_cap) and pd.notna(price) and pd.notna(shares):
                market_cap=price*shares
        except Exception:
            pass
    return price,market_cap,shares

def _ttm_dividend_yield(obj,price):
    if pd.isna(price) or price<=0:
        return np.nan
    try:
        actions=obj.actions
        if actions is None or actions.empty or "Dividends" not in actions.columns:
            return np.nan
        a=actions.copy()
        if not isinstance(a.index,pd.DatetimeIndex):
            return np.nan
        cutoff=pd.Timestamp.now(tz=a.index.tz)-pd.Timedelta(days=365) if a.index.tz is not None else pd.Timestamp.now()-pd.Timedelta(days=365)
        div=float(a.loc[a.index>=cutoff,"Dividends"].sum())
        return div/price if div>0 else 0.0
    except Exception:
        return np.nan

@st.cache_data(ttl=21600,show_spinner=False)
def fetch_fundamental_snapshot(ticker):
    """
    Statement-based secondary engine.
    Ratios are calculated from Yahoo Finance statement feeds rather than Ticker.info.
    Official IDX/company filings remain the verification source.
    """
    try:
        obj=yf.Ticker(f"{ticker}.JK")
    except Exception:
        return {}

    # Try quarterly first, annual as fallback.
    try: qi=obj.quarterly_income_stmt
    except Exception: qi=pd.DataFrame()
    try: qb=obj.quarterly_balance_sheet
    except Exception: qb=pd.DataFrame()
    try: qc=obj.quarterly_cashflow
    except Exception: qc=pd.DataFrame()
    try: ai=obj.income_stmt
    except Exception: ai=pd.DataFrame()
    try: ab=obj.balance_sheet
    except Exception: ab=pd.DataFrame()
    try: ac=obj.cashflow
    except Exception: ac=pd.DataFrame()

    # Normalize newest column first.
    for df in [qi,qb,qc,ai,ab,ac]:
        try:
            if df is not None and not df.empty:
                df.sort_index(axis=1,ascending=False,inplace=True)
        except Exception:
            pass

    # Income statement rows
    q_rev=_statement_row(qi,["Total Revenue","Operating Revenue"])
    q_ni=_statement_row(qi,["Net Income","Net Income Common Stockholders","Net Income Including Noncontrolling Interests"])
    q_op=_statement_row(qi,["Operating Income"])
    q_gp=_statement_row(qi,["Gross Profit"])
    q_ebitda=_statement_row(qi,["EBITDA","Normalized EBITDA"])

    a_rev=_statement_row(ai,["Total Revenue","Operating Revenue"])
    a_ni=_statement_row(ai,["Net Income","Net Income Common Stockholders","Net Income Including Noncontrolling Interests"])
    a_op=_statement_row(ai,["Operating Income"])
    a_gp=_statement_row(ai,["Gross Profit"])
    a_ebitda=_statement_row(ai,["EBITDA","Normalized EBITDA"])

    # Balance sheet rows
    q_equity=_statement_row(qb,["Stockholders Equity","Total Equity Gross Minority Interest","Common Stock Equity"])
    q_assets=_statement_row(qb,["Total Assets"])
    q_debt=_statement_row(qb,["Total Debt"])
    q_cash=_statement_row(qb,["Cash Cash Equivalents And Short Term Investments","Cash And Cash Equivalents","Cash"])
    q_current_assets=_statement_row(qb,["Current Assets","Total Current Assets"])
    q_current_liab=_statement_row(qb,["Current Liabilities","Total Current Liabilities"])

    a_equity=_statement_row(ab,["Stockholders Equity","Total Equity Gross Minority Interest","Common Stock Equity"])
    a_assets=_statement_row(ab,["Total Assets"])
    a_debt=_statement_row(ab,["Total Debt"])
    a_cash=_statement_row(ab,["Cash Cash Equivalents And Short Term Investments","Cash And Cash Equivalents","Cash"])
    a_current_assets=_statement_row(ab,["Current Assets","Total Current Assets"])
    a_current_liab=_statement_row(ab,["Current Liabilities","Total Current Liabilities"])

    # Cash flow
    q_cfo=_statement_row(qc,["Operating Cash Flow","Total Cash From Operating Activities"])
    q_capex=_statement_row(qc,["Capital Expenditure","Capital Expenditures"])
    a_cfo=_statement_row(ac,["Operating Cash Flow","Total Cash From Operating Activities"])
    a_capex=_statement_row(ac,["Capital Expenditure","Capital Expenditures"])

    # TTM where possible, otherwise latest annual
    revenue=_sum_latest(q_rev,4) if q_rev is not None and len(q_rev)>=4 else _latest_value(a_rev)
    net_income=_sum_latest(q_ni,4) if q_ni is not None and len(q_ni)>=4 else _latest_value(a_ni)
    op_income=_sum_latest(q_op,4) if q_op is not None and len(q_op)>=4 else _latest_value(a_op)
    gross_profit=_sum_latest(q_gp,4) if q_gp is not None and len(q_gp)>=4 else _latest_value(a_gp)
    ebitda=_sum_latest(q_ebitda,4) if q_ebitda is not None and len(q_ebitda)>=4 else _latest_value(a_ebitda)
    cfo=_sum_latest(q_cfo,4) if q_cfo is not None and len(q_cfo)>=4 else _latest_value(a_cfo)
    capex=_sum_latest(q_capex,4) if q_capex is not None and len(q_capex)>=4 else _latest_value(a_capex)
    # Capex often negative in statements; FCF = CFO + capex when capex is negative.
    fcf=np.nan if pd.isna(cfo) or pd.isna(capex) else cfo+capex

    equity=_latest_value(q_equity) if q_equity is not None else _latest_value(a_equity)
    assets=_latest_value(q_assets) if q_assets is not None else _latest_value(a_assets)
    debt=_latest_value(q_debt) if q_debt is not None else _latest_value(a_debt)
    cash=_latest_value(q_cash) if q_cash is not None else _latest_value(a_cash)
    current_assets=_latest_value(q_current_assets) if q_current_assets is not None else _latest_value(a_current_assets)
    current_liab=_latest_value(q_current_liab) if q_current_liab is not None else _latest_value(a_current_liab)

    # Growth
    revenue_growth=_yoy_from_quarters(q_rev)
    if pd.isna(revenue_growth):
        revenue_growth=_annual_growth(a_rev)
    earnings_growth=_yoy_from_quarters(q_ni)
    if pd.isna(earnings_growth):
        earnings_growth=_annual_growth(a_ni)

    # Average balance sheet bases for ROE/ROA if enough data
    eq_prev=float(q_equity.iloc[4]) if q_equity is not None and len(q_equity)>=5 else (float(a_equity.iloc[1]) if a_equity is not None and len(a_equity)>=2 else np.nan)
    asset_prev=float(q_assets.iloc[4]) if q_assets is not None and len(q_assets)>=5 else (float(a_assets.iloc[1]) if a_assets is not None and len(a_assets)>=2 else np.nan)
    avg_equity=np.nanmean([equity,eq_prev]) if pd.notna(equity) and pd.notna(eq_prev) else equity
    avg_assets=np.nanmean([assets,asset_prev]) if pd.notna(assets) and pd.notna(asset_prev) else assets

    roe=_safe_div(net_income,avg_equity)
    roa=_safe_div(net_income,avg_assets)
    gross_margin=_safe_div(gross_profit,revenue)
    op_margin=_safe_div(op_income,revenue)
    net_margin=_safe_div(net_income,revenue)
    der=_safe_div(debt,equity)
    current_ratio=_safe_div(current_assets,current_liab)
    net_debt=np.nan if pd.isna(debt) or pd.isna(cash) else debt-cash

    price,market_cap,shares=_get_market_price_and_cap(obj,ticker)
    pe=_safe_div(market_cap,net_income)
    pb=_safe_div(market_cap,equity)
    ev=np.nan if pd.isna(market_cap) else market_cap+(0 if pd.isna(debt) else debt)-(0 if pd.isna(cash) else cash)
    ev_ebitda=_safe_div(ev,ebitda)
    ev_sales=_safe_div(ev,revenue)
    div_yield=_ttm_dividend_yield(obj,price)

    # Metadata is optional only.
    company=ticker; sector="N/A"; industry="N/A"
    try:
        info=obj.info or {}
        company=info.get("longName") or info.get("shortName") or ticker
        sector=info.get("sector") or "N/A"
        industry=info.get("industry") or "N/A"
    except Exception:
        pass

    # Determine basis
    statement_basis="Quarterly/TTM calculated" if q_rev is not None and len(q_rev)>=4 else ("Annual calculated" if a_rev is not None and len(a_rev)>=1 else "Statements unavailable")

    return {
        "company":company,"sector":sector,"industry":industry,
        "statementBasis":statement_basis,
        "marketCap":market_cap,"price":price,
        "revenue":revenue,"netIncome":net_income,"operatingIncome":op_income,"ebitda":ebitda,
        "revenueGrowth":revenue_growth,"earningsGrowth":earnings_growth,
        "returnOnEquity":roe,"returnOnAssets":roa,
        "grossMargins":gross_margin,"operatingMargins":op_margin,"profitMargins":net_margin,
        "totalAssets":assets,"equity":equity,"totalDebt":debt,"totalCash":cash,"netDebt":net_debt,
        "debtToEquity":der,"currentRatio":current_ratio,
        "operatingCashflow":cfo,"capex":capex,"freeCashflow":fcf,
        "trailingPE":pe,"priceToBook":pb,"enterpriseToEbitda":ev_ebitda,"enterpriseToRevenue":ev_sales,
        "dividendYield":div_yield,
        "dataSource":"Yahoo Finance financial statements (secondary) — ratios calculated in dashboard"
    }

def fundamental_interpretation(f):
    if not f:
        return "Data fundamental belum berhasil diperoleh.",[],[]

    positives=[]; risks=[]
    rg=f.get("revenueGrowth",np.nan); eg=f.get("earningsGrowth",np.nan)
    roe=f.get("returnOnEquity",np.nan); pm=f.get("profitMargins",np.nan)
    der=f.get("debtToEquity",np.nan); ocf=f.get("operatingCashflow",np.nan)
    fcf=f.get("freeCashflow",np.nan); pe=f.get("trailingPE",np.nan); pb=f.get("priceToBook",np.nan)

    if pd.notna(rg):
        if rg>0.10: positives.append("Pendapatan tumbuh cukup kuat.")
        elif rg<0: risks.append("Pendapatan sedang menurun.")
    if pd.notna(eg):
        if eg>0.10: positives.append("Laba tumbuh cukup kuat.")
        elif eg<0: risks.append("Laba sedang menurun.")
    if pd.notna(roe):
        if roe>=0.15: positives.append("ROE menunjukkan kemampuan menghasilkan laba atas modal yang baik.")
        elif roe<0.05: risks.append("ROE masih rendah.")
    if pd.notna(pm):
        if pm>0.10: positives.append("Margin laba bersih relatif sehat.")
        elif pm<0: risks.append("Margin laba bersih negatif.")
    if pd.notna(der):
        if der>2: risks.append("Utang relatif tinggi terhadap ekuitas.")
        elif der<1: positives.append("Leverage relatif terkendali.")
    if pd.notna(ocf):
        if ocf>0: positives.append("Arus kas operasi positif.")
        else: risks.append("Arus kas operasi negatif.")
    if pd.notna(fcf) and fcf<0: risks.append("Free cash flow negatif.")
    if pd.notna(pe) and pe<=0: risks.append("PER tidak bermakna/negatif karena laba tidak positif.")
    if pd.notna(pb) and pb>5: risks.append("PBV tinggi; perlu dibandingkan dengan sektor dan historis.")

    if positives and len(positives)>=len(risks)+2:
        summary="Fundamental terlihat cukup sehat dari data statement sekunder yang tersedia."
    elif risks and len(risks)>len(positives):
        summary="Ada beberapa area fundamental yang perlu diperiksa lebih lanjut."
    else:
        summary="Fundamental terlihat campuran; belum cukup kuat untuk simpulan tunggal."
    return summary,positives,risks


def _news_get(d,*path,default=None):
    cur=d
    for key in path:
        if not isinstance(cur,dict) or key not in cur:
            return default
        cur=cur[key]
    return cur

def _first_nonempty(*vals):
    for v in vals:
        if v is not None and v!="" and v!=[]:
            return v
    return None

def _parse_news_item(item):
    """
    Handles both legacy flat yfinance news objects and newer nested `content` objects.
    """
    if not isinstance(item,dict):
        return None

    content=item.get("content") if isinstance(item.get("content"),dict) else item

    title=_first_nonempty(
        content.get("title"),
        item.get("title")
    ) or "Tanpa judul"

    publisher=_first_nonempty(
        _news_get(content,"provider","displayName"),
        content.get("publisher"),
        item.get("publisher"),
        _news_get(item,"provider","displayName")
    ) or "Sumber tidak tercantum"

    # URL structures differ across yfinance versions.
    url=_first_nonempty(
        _news_get(content,"canonicalUrl","url"),
        _news_get(content,"clickThroughUrl","url"),
        content.get("link"),
        item.get("link"),
        _news_get(item,"canonicalUrl","url")
    )

    pubdate=_first_nonempty(
        content.get("pubDate"),
        content.get("displayTime"),
        item.get("pubDate"),
        item.get("providerPublishTime")
    )

    summary=_first_nonempty(
        content.get("summary"),
        content.get("description"),
        item.get("summary")
    ) or ""

    return {
        "title":str(title),
        "publisher":str(publisher),
        "url":url,
        "published":pubdate,
        "summary":str(summary)
    }

def _format_news_date(v):
    if v is None or v=="":
        return "Tanggal tidak tersedia"
    try:
        if isinstance(v,(int,float,np.integer,np.floating)):
            ts=pd.to_datetime(v,unit="s",utc=True).tz_convert("Asia/Jakarta")
        else:
            ts=pd.to_datetime(v,utc=True).tz_convert("Asia/Jakarta")
        return ts.strftime("%d %b %Y %H:%M WIB")
    except Exception:
        return str(v)

POSITIVE_NEWS=[
    "profit rises","profit jumps","profit growth","earnings beat","record profit","revenue growth",
    "wins contract","new contract","awarded contract","dividend","buyback","share buyback",
    "expansion","acquisition approved","strategic partnership","partnership","new plant",
    "upgrade","raises guidance","strong demand","sales growth","debt repayment",
    "laba naik","laba tumbuh","pendapatan naik","pendapatan tumbuh","kontrak baru",
    "menang tender","dividen","buyback","ekspansi","kemitraan","pelunasan utang",
    "target naik","kinerja positif"
]
NEGATIVE_NEWS=[
    "profit falls","profit drops","loss widens","net loss","revenue falls","misses estimates",
    "default","downgrade","lawsuit","investigation","fraud","suspension","bankruptcy",
    "rights issue dilution","share dilution","debt concern","layoffs","recall","fine",
    "laba turun","rugi","kerugian","pendapatan turun","gagal bayar","penurunan peringkat",
    "gugatan","investigasi","fraud","suspensi","pailit","dilusi","utang meningkat",
    "denda","phk"
]

HIGH_MATERIALITY=[
    "acquisition","merger","rights issue","private placement","buyback","dividend","default",
    "bankruptcy","suspension","fraud","investigation","lawsuit","contract","tender",
    "earnings","profit","revenue","guidance","plant","factory","mine","production",
    "acquire","divest","ipo","spin off","stock split",
    "akuisisi","merger","rights issue","penambahan modal","buyback","dividen","gagal bayar",
    "pailit","suspensi","fraud","investigasi","gugatan","kontrak","tender","laba",
    "pendapatan","pabrik","tambang","produksi","divestasi","stock split"
]
MEDIUM_MATERIALITY=[
    "partnership","collaboration","launch","product","market share","capex","expansion",
    "management","director","commissioner","rating","upgrade","downgrade","regulation",
    "policy","tariff","commodity","oil","coal","nickel","gold","interest rate","rupiah",
    "kemitraan","kolaborasi","produk","pangsa pasar","belanja modal","ekspansi",
    "direktur","komisaris","peringkat","regulasi","kebijakan","tarif","komoditas",
    "minyak","batubara","nikel","emas","suku bunga","rupiah"
]

CATEGORY_KEYWORDS={
    "Kinerja keuangan":["earnings","profit","revenue","sales","margin","laba","pendapatan","penjualan"],
    "Aksi korporasi":["acquisition","merger","rights issue","private placement","buyback","dividend","stock split","divest","akuisisi","dividen","penambahan modal","divestasi"],
    "Kontrak / ekspansi":["contract","tender","plant","factory","expansion","partnership","kontrak","pabrik","ekspansi","kemitraan"],
    "Utang / pendanaan":["debt","bond","loan","default","refinancing","utang","obligasi","pinjaman","gagal bayar"],
    "Regulasi / hukum":["regulation","policy","lawsuit","investigation","fine","suspension","regulasi","kebijakan","gugatan","investigasi","denda","suspensi"],
    "Makro / komoditas":["oil","coal","nickel","gold","commodity","rupiah","interest rate","minyak","batubara","nikel","emas","komoditas","suku bunga"]
}

def classify_news_text(title,summary=""):
    text=f"{title} {summary}".lower()

    pos=sum(1 for k in POSITIVE_NEWS if k in text)
    neg=sum(1 for k in NEGATIVE_NEWS if k in text)
    if pos>neg:
        sentiment="Positif"
    elif neg>pos:
        sentiment="Negatif"
    else:
        sentiment="Netral"

    if any(k in text for k in HIGH_MATERIALITY):
        materiality="Tinggi"
    elif any(k in text for k in MEDIUM_MATERIALITY):
        materiality="Sedang"
    else:
        materiality="Rendah"

    category="Umum"
    maxhits=0
    for cat,keys in CATEGORY_KEYWORDS.items():
        hits=sum(1 for k in keys if k in text)
        if hits>maxhits:
            category=cat; maxhits=hits

    if materiality=="Tinggi":
        impact="Berpotensi mengubah pendapatan, laba, arus kas, struktur modal, atau persepsi risiko. Baca isi berita dan cek sumber resmi."
    elif materiality=="Sedang":
        impact="Dapat memengaruhi ekspektasi pasar, tetapi dampaknya perlu dilihat dari skala dan konteks."
    else:
        impact="Kemungkinan lebih bersifat informasi tambahan/noise sampai ada bukti dampak bisnis yang jelas."

    return sentiment,materiality,category,impact

def _company_name_for_news(ticker):
    try:
        obj=yf.Ticker(f"{ticker}.JK")
        info=obj.info or {}
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker

def _google_news_rss(ticker,company_name,count=8):
    """
    Google News RSS fallback. No API key required.
    Query uses ticker + company name + Indonesian stock context.
    """
    query=f'"{ticker}" saham OR "{company_name}"'
    url=(
        "https://news.google.com/rss/search?q="
        + urllib.parse.quote(query)
        + "&hl=id&gl=ID&ceid=ID:id"
    )
    req=urllib.request.Request(
        url,
        headers={"User-Agent":"Mozilla/5.0"}
    )
    with urllib.request.urlopen(req,timeout=12) as resp:
        data=resp.read()

    root=ET.fromstring(data)
    out=[]
    for item in root.findall(".//item")[:count]:
        title=(item.findtext("title") or "").strip()
        link=(item.findtext("link") or "").strip()
        pubdate=(item.findtext("pubDate") or "").strip()
        source_node=item.find("source")
        publisher=(source_node.text or "").strip() if source_node is not None and source_node.text else "Google News"
        description=(item.findtext("description") or "").strip()

        if " - " in title:
            # Google often appends publisher after " - "
            headline,maybe_pub=title.rsplit(" - ",1)
            if maybe_pub and len(maybe_pub)<80:
                title=headline.strip()
                if publisher=="Google News":
                    publisher=maybe_pub.strip()

        out.append({
            "title":title or "Tanpa judul",
            "publisher":publisher,
            "url":link,
            "published":pubdate,
            "summary":re.sub(r"<[^>]+>"," ",description)
        })
    return out


def _normalize_text(s):
    s=str(s or "").lower()
    s=re.sub(r"[^a-z0-9\s]"," ",s)
    s=re.sub(r"\s+"," ",s).strip()
    return s

def _company_aliases(ticker,company_name):
    """
    Build conservative aliases from ticker/company name.
    Example: PT Erajaya Swasembada Tbk -> erajaya swasembada, erajaya, ERAA.
    """
    aliases={ticker.lower()}
    name=_normalize_text(company_name)
    stop={"pt","tbk","persero","perseroan","terbatas","indonesia","the","company","co","ltd"}
    tokens=[x for x in name.split() if x not in stop and len(x)>=4]

    if tokens:
        aliases.add(" ".join(tokens))
        # first distinctive token is often brand/company family name
        aliases.add(tokens[0])
        if len(tokens)>=2:
            aliases.add(" ".join(tokens[:2]))

    # Avoid aliases that are too generic
    bad={"global","group","holding","energy","digital","technology","international","utama","jaya","sejahtera"}
    aliases={a for a in aliases if len(a)>=4 and a not in bad}
    return sorted(aliases,key=len,reverse=True)

def _direct_relevance(ticker,company_name,title,summary,url=""):
    """
    Strict entity relevance.
    High: ticker or multi-word company alias in title.
    Medium: direct alias appears in summary, or distinctive company token appears in title.
    Low: no direct company evidence -> excluded.
    """
    aliases=_company_aliases(ticker,company_name)
    title_n=_normalize_text(title)
    summary_n=_normalize_text(summary)
    url_n=_normalize_text(url)

    ticker_l=ticker.lower()

    # Exact ticker in title, with word boundaries.
    if re.search(rf"\b{re.escape(ticker_l)}\b",title_n):
        return "Tinggi",3,"Ticker disebut langsung pada judul."

    # Multi-word company aliases in title are strong evidence.
    for a in aliases:
        if " " in a and a in title_n:
            return "Tinggi",3,"Nama emiten disebut langsung pada judul."

    # Distinctive single-token aliases in title.
    for a in aliases:
        if " " not in a and len(a)>=6 and re.search(rf"\b{re.escape(a)}\b",title_n):
            return "Tinggi",3,"Nama/brand emiten disebut langsung pada judul."

    # Summary direct mention.
    if re.search(rf"\b{re.escape(ticker_l)}\b",summary_n):
        return "Sedang",2,"Ticker disebut pada ringkasan berita."

    for a in aliases:
        if len(a)>=6 and a in summary_n:
            return "Sedang",2,"Nama emiten disebut pada ringkasan berita."

    # URL can occasionally contain ticker/company slug, but weaker.
    for a in aliases:
        compact=a.replace(" ","")
        if len(compact)>=6 and compact in url_n.replace(" ",""):
            return "Sedang",2,"Tautan berita menunjukkan keterkaitan dengan emiten."

    return "Rendah",0,"Tidak ada bukti hubungan langsung dengan emiten."

def _source_tier(publisher,url=""):
    p=_normalize_text(publisher)
    u=str(url or "").lower()

    # Tier 1: official / regulatory when recognizable.
    official_tokens=[
        "idx","indonesia stock exchange","bursa efek indonesia","ojk",
        "ksei","official","investor relations"
    ]
    official_domains=["idx.co.id","ojk.go.id","ksei.co.id"]
    if any(t in p for t in official_tokens) or any(d in u for d in official_domains):
        return "Tier 1 — Resmi",3

    # Tier 2: established financial/news outlets (non-exhaustive).
    tier2_tokens=[
        "reuters","bloomberg","antara","kontan","bisnis","cnbc indonesia",
        "tempo","kompas","detikfinance","investor daily","katadata",
        "the jakarta post","nikkei","financial times"
    ]
    if any(t in p for t in tier2_tokens):
        return "Tier 2 — Media kredibel",2

    return "Tier 3 — Agregator/media lain",1

def _materiality_score(label):
    return {"Tinggi":3,"Sedang":2,"Rendah":1}.get(label,1)

def _sentiment_score(label):
    return {"Positif":1,"Netral":0,"Negatif":-1}.get(label,0)

@st.cache_data(ttl=1800,show_spinner=False)
def fetch_news_snapshot(ticker,count=8):
    """
    Relevance-filtered hybrid engine:
    1) Yahoo/yfinance ticker news
    2) yfinance Search
    3) Google News RSS Indonesia
    4) STRICT direct-entity relevance filter
    Only Medium/High relevance survives.
    """
    items=[]
    company_name=_company_name_for_news(ticker)

    # 1. Yahoo/yfinance ticker news
    try:
        obj=yf.Ticker(f"{ticker}.JK")
        raw=obj.get_news(count=max(count*2,12),tab="news")
        if raw:
            items.extend([("Yahoo Finance/yfinance",x) for x in raw])
    except Exception:
        pass

    # 2. Search fallback. Use quoted company/ticker to improve precision.
    try:
        search=yf.Search(
            f'"{ticker}" OR "{company_name}"',
            max_results=5,
            news_count=max(count*2,12)
        )
        extra=getattr(search,"news",[]) or []
        items.extend([("Yahoo Search",x) for x in extra])
    except Exception:
        pass

    # 3. Google News RSS always queried because it is often better for IDX local coverage.
    try:
        rss_items=_google_news_rss(ticker,company_name,count=max(count*3,20))
        items.extend([("Google News RSS",x) for x in rss_items])
    except Exception:
        pass

    parsed=[]
    seen=set()

    for engine,item in items:
        if engine=="Google News RSS":
            p=item
        else:
            p=_parse_news_item(item)
        if not p:
            continue

        title=p.get("title","")
        summary=p.get("summary","")
        url=p.get("url") or ""

        relevance,rel_score,rel_reason=_direct_relevance(
            ticker,company_name,title,summary,url
        )

        # Hard filter: low relevance never enters dashboard/sentiment.
        if rel_score<2:
            continue

        key=_normalize_text(title)
        if not key or key in seen:
            continue
        seen.add(key)

        sentiment,materiality,category,impact=classify_news_text(title,summary)
        tier,tier_score=_source_tier(p.get("publisher",""),url)

        p.update({
            "sentiment":sentiment,
            "materiality":materiality,
            "category":category,
            "impact":impact,
            "dateText":_format_news_date(p.get("published")),
            "engineSource":engine,
            "relevance":relevance,
            "relevanceScore":rel_score,
            "relevanceReason":rel_reason,
            "sourceTier":tier,
            "sourceTierScore":tier_score
        })
        parsed.append(p)

    # Sort priority:
    # 1. direct relevance
    # 2. source quality
    # 3. materiality
    parsed.sort(
        key=lambda n:(
            n.get("relevanceScore",0),
            n.get("sourceTierScore",0),
            _materiality_score(n.get("materiality"))
        ),
        reverse=True
    )

    return parsed[:count]

def news_overall_summary(news):
    if not news:
        return "Belum ada berita yang lolos filter relevansi emiten.", "Netral"

    weights={"Tinggi":3,"Sedang":2,"Rendah":1}
    relweights={"Tinggi":1.5,"Sedang":1.0}
    scores={"Positif":1,"Netral":0,"Negatif":-1}
    total=0; denom=0
    high_material=[]

    for n in news:
        w=weights.get(n["materiality"],1)*relweights.get(n.get("relevance"),1)
        total+=scores.get(n["sentiment"],0)*w
        denom+=w
        if n["materiality"]=="Tinggi" and n.get("relevance") in ("Tinggi","Sedang"):
            high_material.append(n)

    avg=total/denom if denom else 0
    if avg>0.20:
        tone="Cenderung Positif"
    elif avg<-0.20:
        tone="Cenderung Negatif"
    else:
        tone="Campuran / Netral"

    if high_material:
        summary=(
            f"Sentimen berita relevan {tone.lower()}. "
            f"Ada {len(high_material)} berita material tinggi yang terkait langsung dengan emiten."
        )
    else:
        summary=(
            f"Sentimen berita relevan {tone.lower()}. "
            "Belum terlihat berita material tinggi yang terkait langsung dengan emiten."
        )
    return summary,tone


def _fundamental_card_score(f):
    if not f:
        return 0,"Data fundamental terbatas",[],["Data fundamental belum cukup."]

    score=0; pos=[]; risk=[]

    rg=f.get("revenueGrowth",np.nan)
    eg=f.get("earningsGrowth",np.nan)
    roe=f.get("returnOnEquity",np.nan)
    pm=f.get("profitMargins",np.nan)
    der=f.get("debtToEquity",np.nan)
    ocf=f.get("operatingCashflow",np.nan)
    fcf=f.get("freeCashflow",np.nan)
    pe=f.get("trailingPE",np.nan)
    pb=f.get("priceToBook",np.nan)

    if pd.notna(rg):
        if rg>0.10: score+=1; pos.append("Pendapatan tumbuh >10%.")
        elif rg<0: score-=1; risk.append("Pendapatan menurun.")
    if pd.notna(eg):
        if eg>0.10: score+=1; pos.append("Laba tumbuh >10%.")
        elif eg<0: score-=1; risk.append("Laba menurun.")
    if pd.notna(roe):
        if roe>=0.15: score+=1; pos.append("ROE ≥15%.")
        elif roe<0.05: score-=1; risk.append("ROE rendah.")
    if pd.notna(pm) and pm<0:
        score-=1; risk.append("Margin laba bersih negatif.")
    if pd.notna(der):
        if der<1: score+=1; pos.append("DER <1x.")
        elif der>2: score-=1; risk.append("DER >2x.")
    if pd.notna(ocf):
        if ocf>0: score+=1; pos.append("Arus kas operasi positif.")
        else: score-=1; risk.append("Arus kas operasi negatif.")
    if pd.notna(fcf) and fcf<0:
        risk.append("Free cash flow negatif.")
    if pd.notna(pe) and pe>0 and pe<15:
        pos.append("PER relatif rendah secara absolut; tetap perlu dibandingkan sektor.")
    if pd.notna(pb) and pb>5:
        risk.append("PBV tinggi secara absolut.")

    label="Kuat" if score>=4 else "Cukup" if score>=2 else "Campuran" if score>=0 else "Lemah"
    return score,label,pos,risk

def _news_card_score(news):
    if not news:
        return 0,"Belum ada berita relevan",[],["Belum ada berita relevan yang lolos filter."]

    score=0; pos=[]; risk=[]
    weights={"Tinggi":2,"Sedang":1,"Rendah":0}
    for n in news:
        w=weights.get(n.get("materiality"),0)
        if n.get("sentiment")=="Positif":
            score+=w
            if w>0: pos.append(f"{n.get('materiality')} material: {n.get('title','')}")
        elif n.get("sentiment")=="Negatif":
            score-=w
            if w>0: risk.append(f"{n.get('materiality')} material: {n.get('title','')}")

    label="Positif" if score>=2 else "Negatif" if score<=-2 else "Netral/Campuran"
    return score,label,pos[:3],risk[:3]

def _technical_card_score(last,trend,plan):
    score=0; pos=[]; risk=[]
    if trend in ("Naik kuat","Cenderung naik"):
        score+=2; pos.append(f"Arah harga: {trend}.")
    elif trend in ("Turun kuat","Cenderung turun"):
        score-=2; risk.append(f"Arah harga: {trend}.")
    else:
        pos.append("Harga masih netral/sideways.")

    rsi=last.get("RSI14",np.nan)
    if pd.notna(rsi):
        if 50<=rsi<70:
            score+=1; pos.append("Momentum positif.")
        elif rsi>=70:
            risk.append("Momentum sudah cukup panas.")
        elif rsi<40:
            score-=1; risk.append("Momentum lemah.")

    rvol=last.get("RVOL20",last.get("VolRatio",np.nan))
    if pd.notna(rvol) and rvol>=1.5:
        score+=1; pos.append("Volume sangat kuat dibanding rata-rata.")
    elif pd.notna(rvol) and rvol<0.75:
        risk.append("Volume relatif sepi.")

    if plan:
        rec=plan.get("recommended")
        if rec in ("Entry Konservatif","Breakout Entry"):
            score+=1; pos.append(f"Entry plan: {rec}.")
        elif rec=="Tunggu":
            risk.append("Entry plan masih: Tunggu.")

    label="Positif" if score>=3 else "Netral" if score>=0 else "Negatif"
    return score,label,pos,risk

def _ownership_card_score(delta,pctile,coverage):
    score=0; pos=[]; risk=[]
    if pd.isna(delta):
        return 0,"Data terbatas",[],["Perubahan ownership belum tersedia."]

    if delta<0:
        score+=1; pos.append("Porsi kepemilikan ritel berkurang.")
        if pd.notna(pctile) and pctile>=.90:
            score+=2; pos.append("Penurunan ritel termasuk 10% terbesar.")
        elif pd.notna(pctile) and pctile>=.75:
            score+=1; pos.append("Penurunan ritel termasuk 25% terbesar.")
    elif delta>0:
        score-=1; risk.append("Porsi kepemilikan ritel meningkat.")

    if pd.notna(coverage):
        if coverage>=.95:
            pos.append("Coverage ownership tinggi.")
        elif coverage<.75:
            score-=1; risk.append("Coverage ownership rendah.")

    label="Mendukung" if score>=2 else "Netral" if score>=0 else "Kurang mendukung"
    return score,label,pos,risk

def _market_card_score(mreg):
    if mreg=="Pasar sehat":
        return 1,"Mendukung",["Market regime sehat."],[]
    if mreg=="Pasar lemah":
        return -2,"Kurang mendukung",[],["Market regime lemah."]
    return 0,"Netral",["Market regime netral."],[]

def _final_research_verdict(fund_score,own_score,tech_score,market_score,news_score):
    total=fund_score+own_score+tech_score+market_score+news_score
    if total>=7:
        return "Menarik untuk diteliti lebih lanjut",total
    if total>=3:
        return "Perlu dipantau",total
    if total>=0:
        return "Netral / selektif",total
    return "Waspada / tunggu",total

def _watchlist_items(plan,trend,news_risk,fund_risk,own_risk,mreg):
    items=[]
    if plan:
        rec=plan.get("recommended")
        if rec=="Tunggu Breakout":
            items.append(f"Tunggu harga menembus resistance {rupiah(plan.get('resistance'))} dengan volume kuat.")
        elif rec=="Entry Konservatif":
            items.append("Pantau apakah harga bertahan di atas rata-rata 20/50 hari.")
        elif rec=="Entry Agresif":
            items.append("Pantau apakah support bertahan tanpa breakdown.")
        elif rec=="Tunggu":
            items.append("Belum ada setup entry yang cukup kuat; tunggu konfirmasi.")

    if trend=="Netral / sideways":
        items.append("Tunggu arah harga menjadi lebih jelas.")
    if mreg=="Pasar netral":
        items.append("Pasar belum benar-benar kuat; pilih saham secara selektif.")
    elif mreg=="Pasar lemah":
        items.append("Prioritaskan manajemen risiko karena market regime lemah.")

    if own_risk:
        items.append("Pantau snapshot ownership berikutnya.")
    if fund_risk:
        items.append("Verifikasi area fundamental yang lemah pada laporan resmi terbaru.")
    if news_risk:
        items.append("Baca berita negatif/material tinggi dan cek keterbukaan IDX/emiten.")

    # deduplicate while preserving order
    seen=set(); out=[]
    for x in items:
        if x not in seen:
            out.append(x); seen.add(x)
    return out[:6]


def _portfolio_return_matrix(price_df,tickers,lookback=120):
    frames=[]
    for t in tickers:
        g=price_df[price_df["Ticker"]==t].sort_values("Date")[["Date","Close"]].copy()
        if g.empty:
            continue
        g=g.tail(lookback+5)
        g[t]=g["Close"].pct_change()
        frames.append(g[["Date",t]])
    if not frames:
        return pd.DataFrame()
    out=frames[0]
    for f in frames[1:]:
        out=out.merge(f,on="Date",how="outer")
    return out.sort_values("Date").set_index("Date")

def _portfolio_corr_summary(corr):
    if corr is None or corr.empty or len(corr.columns)<2:
        return np.nan,np.nan,None
    vals=[]
    pair=None
    maxcorr=-9
    cols=list(corr.columns)
    for i in range(len(cols)):
        for j in range(i+1,len(cols)):
            v=corr.iloc[i,j]
            if pd.notna(v):
                vals.append(v)
                if v>maxcorr:
                    maxcorr=v
                    pair=(cols[i],cols[j])
    avg=np.mean(vals) if vals else np.nan
    return avg,maxcorr,pair

def _correlation_label(v):
    if pd.isna(v): return "Belum cukup data"
    if v>=0.75: return "Sangat tinggi"
    if v>=0.50: return "Tinggi"
    if v>=0.25: return "Sedang"
    if v>=0: return "Rendah"
    return "Negatif / diversifikasi baik"

def _portfolio_position_snapshot(price,own,ticker,mreg_raw,rconf,mreg):
    p=price[price["Ticker"]==ticker].sort_values("Date")
    if len(p)<20:
        return None
    tf=technical_frame(p)
    last=tf.iloc[-1]
    trend=trend_label(last)

    ot=own[(own["Ticker"]==ticker)&own["RetailPct"].notna()].sort_values("Date")
    if not ot.empty:
        lo=ot.iloc[-1]
        delta=lo["DeltaRetail"]; pctile=lo["OwnershipPctile"]; cov=lo["Coverage"]
        owntext=ownership_text(delta,pctile)
        hasown=True
    else:
        delta=pctile=cov=np.nan
        owntext="Ownership belum tersedia"
        hasown=False

    lab,score,_,risk=evidence(
        delta,pctile,last["VolRatio"],mreg_raw,rconf,cov,trend,hasown
    )
    override="Ya" if any("pasar" in x.lower() and "lemah" in x.lower() for x in risk) else "Tidak"
    status=action_bucket(lab,override,trend)

    return {
        "Ticker":ticker,
        "Close":last["Close"],
        "Ret20D":last["Ret20D"],
        "Ret60D":last["Ret60D"],
        "Trend":trend,
        "RSI14":last["RSI14"],
        "VolRatio":last["VolRatio"],
        "OwnershipText":owntext,
        "Coverage":cov,
        "Status":status,
        "Evidence":lab,
        "EvidenceScore":score
    }

def _portfolio_risk_label(vol_ann):
    if pd.isna(vol_ann): return "Belum cukup data"
    if vol_ann<0.20: return "Relatif rendah"
    if vol_ann<0.35: return "Sedang"
    if vol_ann<0.50: return "Tinggi"
    return "Sangat tinggi"

def _position_priority(row,weight):
    reasons=[]
    score=0

    if row["Status"]=="Hindari dulu":
        score+=3; reasons.append("evidence posisi lemah")
    elif row["Status"]=="Watchlist":
        score+=1; reasons.append("masih butuh konfirmasi")

    if row["Trend"] in ("Cenderung turun","Turun kuat"):
        score+=2; reasons.append("arah harga lemah")

    if pd.notna(row["Ret20D"]) and row["Ret20D"]<-0.10:
        score+=1; reasons.append("turun >10% dalam 20 hari")

    if weight>=0.20:
        score+=2; reasons.append("bobot portofolio besar")
    elif weight>=0.12:
        score+=1; reasons.append("bobot cukup besar")

    if pd.notna(row["Coverage"]) and row["Coverage"]<0.75:
        score+=1; reasons.append("coverage ownership rendah")

    label="Prioritas tinggi" if score>=4 else "Perlu dipantau" if score>=2 else "Normal"
    return label,score,", ".join(reasons) if reasons else "tidak ada peringatan utama"

def _portfolio_summary_text(sector_max,avg_corr,vol_ann,weak_weight):
    notes=[]
    if sector_max>=0.40:
        notes.append("konsentrasi sektor tinggi")
    elif sector_max>=0.25:
        notes.append("konsentrasi sektor perlu dipantau")
    else:
        notes.append("konsentrasi sektor relatif tersebar")

    if pd.notna(avg_corr):
        if avg_corr>=0.50:
            notes.append("korelasi antar-saham tinggi")
        elif avg_corr>=0.25:
            notes.append("korelasi sedang")
        else:
            notes.append("diversifikasi korelasi cukup baik")

    if pd.notna(vol_ann):
        notes.append(f"volatilitas portofolio {_portfolio_risk_label(vol_ann).lower()}")

    if weak_weight>=0.25:
        notes.append("porsi saham dengan evidence lemah cukup besar")
    elif weak_weight>0:
        notes.append("ada sebagian posisi dengan evidence lemah")
    else:
        notes.append("tidak ada bobot besar pada posisi evidence lemah")

    return "; ".join(notes).capitalize()+"."

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

tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8=st.tabs(["⭐ Pilihan Hari Ini","🔎 Semua Saham","📊 Detail Saham","🏢 Fundamental","📰 Berita & Sentimen","🧾 Final Research Card","💼 Portfolio","🌐 Kondisi Pasar"])

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
        st.caption(
            f"Perusahaan: **{f['company']}** · Sektor: **{f['sector']}** · Industri: **{f['industry']}**"
        )
        st.caption(
            f"Basis perhitungan: **{f['statementBasis']}** · "
            f"Sumber otomatis: **{f['dataSource']}**"
        )

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
        der=f["debtToEquity"]
        h1.metric("DER / Utang : Ekuitas","N/A" if pd.isna(der) else f"{der:.2f}x")
        h1.caption("Semakin tinggi, semakin besar ketergantungan pada utang. Tidak cocok dibandingkan lintas sektor tanpa konteks.")
        h2.metric("Kas",fmt_number(f["totalCash"]))
        h2.caption("Kas dan setara kas yang dilaporkan oleh sumber sekunder.")
        h3.metric("Arus kas operasi",fmt_number(f["operatingCashflow"]))
        h3.caption("Kas yang dihasilkan oleh operasi utama perusahaan.")
        h4.metric("Free cash flow",fmt_number(f["freeCashflow"]))
        h4.caption("Kas setelah kebutuhan investasi utama; negatif tidak selalu buruk jika perusahaan sedang ekspansi.")

        with st.expander("Lihat angka dasar yang dipakai menghitung rasio"):
            b1,b2,b3,b4=st.columns(4)
            b1.metric("Pendapatan (TTM/annual)",fmt_number(f["revenue"]))
            b2.metric("Laba bersih (TTM/annual)",fmt_number(f["netIncome"]))
            b3.metric("Ekuitas terakhir",fmt_number(f["equity"]))
            b4.metric("Total aset",fmt_number(f["totalAssets"]))
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Total utang",fmt_number(f["totalDebt"]))
            c2.metric("Kas",fmt_number(f["totalCash"]))
            c3.metric("Net debt",fmt_number(f["netDebt"]))
            c4.metric("Capex",fmt_number(f["capex"]))

        st.markdown("### 3. Valuasi")
        v1,v2,v3,v4,v5=st.columns(5)
        v1.metric("PER","N/A" if pd.isna(f["trailingPE"]) else f"{f['trailingPE']:.1f}x")
        v1.caption("Harga saham dibanding laba. PER rendah belum otomatis murah.")
        v2.metric("PBV","N/A" if pd.isna(f["priceToBook"]) else f"{f['priceToBook']:.1f}x")
        v2.caption("Harga dibanding nilai buku. Sangat tergantung karakter sektor.")
        v3.metric("EV/EBITDA","N/A" if pd.isna(f["enterpriseToEbitda"]) else f"{f['enterpriseToEbitda']:.1f}x")
        v3.caption("Valuasi perusahaan terhadap EBITDA; lebih relevan untuk bisnis non-keuangan.")
        v4.metric("EV/Sales","N/A" if pd.isna(f["enterpriseToRevenue"]) else f"{f['enterpriseToRevenue']:.1f}x")
        v4.caption("Nilai perusahaan dibanding pendapatan. Berguna untuk bisnis yang laba/EBITDA-nya masih berubah-ubah.")
        v5.metric("Dividend yield",fmt_percent(f["dividendYield"]))
        v5.caption("Dividen 12 bulan terakhir dibanding harga terakhir, jika data tersedia.")

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

        st.markdown("### Kualitas data")
        completeness_fields=[
            f.get("revenue"),f.get("netIncome"),f.get("equity"),f.get("totalDebt"),
            f.get("operatingCashflow"),f.get("trailingPE"),f.get("priceToBook")
        ]
        available_count=sum(pd.notna(x) for x in completeness_fields)
        completeness=available_count/len(completeness_fields)
        if completeness>=0.85:
            st.success(f"**Cukup lengkap ({completeness:.0%})** — sebagian besar angka inti berhasil dihitung.")
        elif completeness>=0.50:
            st.warning(f"**Sebagian tersedia ({completeness:.0%})** — gunakan dengan verifikasi tambahan.")
        else:
            st.error(f"**Data terbatas ({completeness:.0%})** — jangan gunakan untuk simpulan fundamental.")

        st.caption(
            "Status 'cukup lengkap' tidak berarti data sudah diverifikasi resmi. "
            "Angka tetap berasal dari statement feed Yahoo Finance dan perlu dicocokkan dengan laporan IDX/emiten terbaru."
        )

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
    st.subheader("Berita & Sentimen")
    st.write(
        "Bagian ini membantu menjawab: **apa yang sedang terjadi pada emiten, dan apakah beritanya benar-benar material?** "
        "Berita yang tidak menyebut ticker/nama emiten secara langsung akan **dibuang** agar hasil tidak tercampur berita global yang tidak relevan. "
        "Klasifikasi sentimen tetap bersifat **indikatif**, bukan pengganti membaca artikel."
    )

    nticker=st.selectbox(
        "Pilih saham untuk berita",
        full_universe,
        index=full_universe.index("ERAA") if "ERAA" in full_universe else 0,
        key="news_ticker"
    )

    with st.spinner(f"Mengambil berita terbaru {nticker}..."):
        news=fetch_news_snapshot(nticker,8)

    nsum,ntone=news_overall_summary(news)

    if ntone=="Cenderung Positif":
        st.success(f"**Ringkasan sentimen:** {nsum}")
    elif ntone=="Cenderung Negatif":
        st.error(f"**Ringkasan sentimen:** {nsum}")
    else:
        st.info(f"**Ringkasan sentimen:** {nsum}")

    st.caption(
        "Sumber otomatis: Yahoo Finance/yfinance, dengan fallback Google News RSS Indonesia. "
        "Keduanya merupakan sumber sekunder. Untuk aksi korporasi, laporan keuangan, transaksi material, "
        "suspensi, atau isu hukum, cek keterbukaan informasi IDX/emiten sebagai sumber utama."
    )

    if not news:
        st.warning(
            "Belum ada berita yang lolos **filter relevansi emiten**. "
            "Ini lebih baik daripada menampilkan berita yang tidak terkait. "
            "Gunakan keterbukaan IDX/emiten untuk pengecekan manual jika diperlukan."
        )
    else:
        sources=sorted(set(n.get("engineSource","Secondary") for n in news))
        st.caption("Engine berita aktif: **" + " + ".join(sources) + "**")
        # KPIs
        k1,k2,k3,k4,k5=st.columns(5)
        k1.metric("Berita relevan",len(news))
        k2.metric("Relevansi tinggi",sum(n.get("relevance")=="Tinggi" for n in news))
        k3.metric("Positif",sum(n["sentiment"]=="Positif" for n in news))
        k4.metric("Negatif",sum(n["sentiment"]=="Negatif" for n in news))
        k5.metric("Materialitas tinggi",sum(n["materiality"]=="Tinggi" for n in news))

        st.markdown("### Berita terbaru")
        for n in news:
            if n["sentiment"]=="Positif":
                icon="🟢"
            elif n["sentiment"]=="Negatif":
                icon="🔴"
            else:
                icon="⚪"

            mat_icon={"Tinggi":"🔥","Sedang":"🟡","Rendah":"⚪"}.get(n["materiality"],"⚪")

            with st.container(border=True):
                c1,c2=st.columns([4,1.25])
                with c1:
                    st.markdown(f"#### {icon} {n['title']}")
                    st.caption(
                        f"{n['publisher']} · {n['dateText']} · via {n.get('engineSource','feed sekunder')} · "
                        f"{n.get('sourceTier','Tier 3')}"
                    )
                    if n["summary"]:
                        st.write(n["summary"][:500])
                    if n["url"]:
                        st.link_button("Buka berita",n["url"])
                with c2:
                    st.write(f"**Relevansi:** {n.get('relevance','N/A')}")
                    st.write(f"**Sentimen:** {n['sentiment']}")
                    st.write(f"**Materialitas:** {mat_icon} {n['materiality']}")
                    st.write(f"**Kategori:** {n['category']}")

                st.write(f"**Mengapa berita ini dianggap terkait?** {n.get('relevanceReason','')}")
                st.write(f"**Mengapa penting?** {n['impact']}")

        with st.expander("📘 Cara membaca sentimen berita"):
            st.write("**Positif / Netral / Negatif** adalah klasifikasi otomatis berbasis kata dan konteks pendek pada judul/ringkasan. Ini bukan analisis isi penuh.")
            st.write("**Materialitas Tinggi** berarti topiknya berpotensi menyentuh laba, pendapatan, arus kas, struktur modal, kontrak besar, aksi korporasi, atau risiko hukum/regulasi.")
            st.write("**Materialitas Sedang** biasanya berkaitan dengan ekspansi, kemitraan, perubahan manajemen, komoditas, atau kebijakan yang dampaknya belum langsung.")
            st.write("**Materialitas Rendah** berarti belum ada indikasi kuat bahwa berita tersebut mengubah tesis investasi; bisa jadi hanya noise.")
            st.write("**Relevansi Tinggi** berarti ticker/nama emiten muncul langsung pada judul. **Relevansi Sedang** berarti hubungan langsung muncul pada ringkasan/tautan.")
            st.write("Berita dengan **Relevansi Rendah tidak ditampilkan dan tidak masuk perhitungan sentimen**.")
            st.write("Tier 1 = sumber resmi/regulator; Tier 2 = media kredibel; Tier 3 = agregator/media lain.")
            st.write("Jika headline tampak penting, selalu buka artikelnya dan cek keterbukaan informasi resmi sebelum menarik simpulan.")

with tab6:
    st.subheader("Final Research Card")
    st.write(
        "Satu halaman ringkas untuk menjawab: **bisnisnya bagaimana, ownership-nya bagaimana, "
        "harga sedang seperti apa, berita terbaru apa, apa risikonya, dan apa yang perlu ditunggu?**"
    )

    rticker=st.selectbox(
        "Pilih saham untuk Final Research Card",
        full_universe,
        index=full_universe.index("ERAA") if "ERAA" in full_universe else 0,
        key="final_card_ticker"
    )

    with st.spinner(f"Menyusun research card {rticker}..."):
        rp,rp_status=get_price_history(price,rticker)
        rf=fetch_fundamental_snapshot(rticker)
        rn=fetch_news_snapshot(rticker,8)

    rot=own[(own["Ticker"]==rticker)&own["RetailPct"].notna()].sort_values("Date")
    rhasown=not rot.empty

    if rp.empty:
        st.error("Data harga belum tersedia, sehingga Final Research Card belum dapat disusun lengkap.")
    else:
        rtf=technical_frame(rp)
        rlast=rtf.iloc[-1]
        rtrend=trend_label(rlast)

        if rhasown:
            rlo=rot.iloc[-1]
            rdelta=rlo["DeltaRetail"]
            rpctile=rlo["OwnershipPctile"]
            rcov=rlo["Coverage"]
        else:
            rdelta=rpctile=rcov=np.nan

        # preliminary status for trading plan
        rlab,rscore,_,_=evidence(
            rdelta,rpctile,rlast["VolRatio"],
            mreg_raw,rconf,rcov,rtrend,rhasown
        )
        roverride="Ya" if mreg=="Pasar lemah" else "Tidak"
        rstatus=action_bucket(rlab,roverride,rtrend)
        rplan=trading_plan(rtf,rstatus,rtrend,mreg) if len(rtf)>=60 else None

        fscore,flabel,fpos,frisk=_fundamental_card_score(rf)
        oscore,olabel,opos,orisk=_ownership_card_score(rdelta,rpctile,rcov)
        tscore,tlabel,tpos,trisk=_technical_card_score(rlast,rtrend,rplan)
        mscore,mlabel,mpos,mrisk=_market_card_score(mreg)
        nscore,nlabel,npos,nrisk=_news_card_score(rn)

        verdict,total=_final_research_verdict(fscore,oscore,tscore,mscore,nscore)

        if verdict.startswith("Menarik"):
            st.success(f"### {rticker}: {verdict}")
        elif verdict=="Perlu dipantau":
            st.warning(f"### {rticker}: {verdict}")
        elif verdict.startswith("Waspada"):
            st.error(f"### {rticker}: {verdict}")
        else:
            st.info(f"### {rticker}: {verdict}")

        st.caption(
            f"Evidence score gabungan: **{total:+d}**. "
            "Skor ini hanya alat ringkas untuk menyatukan bukti, bukan probabilitas keuntungan."
        )

        # Snapshot cards
        s1,s2,s3,s4,s5=st.columns(5)
        s1.metric("Fundamental",flabel)
        s2.metric("Ownership",olabel)
        s3.metric("Technical",tlabel)
        s4.metric("Market",mlabel)
        s5.metric("News",nlabel)

        st.markdown("### Ringkasan 5 Layer")
        c1,c2=st.columns(2)

        with c1:
            with st.container(border=True):
                st.markdown("#### 🏢 Fundamental")
                if fpos:
                    for x in fpos[:4]: st.write("✅",x)
                if frisk:
                    for x in frisk[:3]: st.write("⚠️",x)
                if not fpos and not frisk:
                    st.write("Data fundamental belum cukup.")

            with st.container(border=True):
                st.markdown("#### 👥 Ownership")
                if opos:
                    for x in opos: st.write("✅",x)
                if orisk:
                    for x in orisk: st.write("⚠️",x)

            with st.container(border=True):
                st.markdown("#### 🌐 Kondisi Pasar")
                for x in mpos: st.write("✅",x)
                for x in mrisk: st.write("⚠️",x)

        with c2:
            with st.container(border=True):
                st.markdown("#### 📈 Technical")
                for x in tpos[:4]: st.write("✅",x)
                for x in trisk[:3]: st.write("⚠️",x)

            with st.container(border=True):
                st.markdown("#### 📰 Berita")
                if rn:
                    st.write(f"Sentimen relevan: **{nlabel}**")
                    for x in npos[:2]: st.write("✅",x)
                    for x in nrisk[:2]: st.write("⚠️",x)
                else:
                    st.write("Belum ada berita relevan yang lolos filter.")

        st.markdown("### Trading Plan")
        if rplan:
            tp1,tp2,tp3,tp4=st.columns(4)
            tp1.metric("Skenario",rplan["recommended"])
            tp2.metric("Support",rupiah(rplan["support"]))
            tp3.metric("Batas batal",rupiah(rplan["invalidation"]))
            tp4.metric("Resistance",rupiah(rplan["resistance"]))

            e1,e2,e3=st.columns(3)
            e1.metric("Entry Agresif",f"{rupiah(rplan['aggressive_low'])} – {rupiah(rplan['aggressive_high'])}")
            e2.metric("Entry Konservatif",f"{rupiah(rplan['conservative_low'])} – {rupiah(rplan['conservative_high'])}")
            e3.metric("Breakout Entry",f"{rupiah(rplan['breakout_low'])} – {rupiah(rplan['breakout_high'])}")

            st.write(f"**Alasan skenario:** {rplan['reason']}")
        else:
            st.warning("Histori harga belum cukup untuk menyusun trading plan.")

        st.markdown("### Risiko utama")
        combined_risks=[]
        combined_risks.extend(frisk[:3])
        combined_risks.extend(orisk[:2])
        combined_risks.extend(trisk[:3])
        combined_risks.extend(mrisk[:2])
        combined_risks.extend(nrisk[:2])

        if combined_risks:
            for x in combined_risks[:8]:
                st.write("⚠️",x)
        else:
            st.write("Belum ada risiko dominan dari model, tetapi risiko pasar tetap ada.")

        st.markdown("### Hal yang perlu dipantau berikutnya")
        watch=_watchlist_items(rplan,rtrend,nrisk,frisk,orisk,mreg)
        for x in watch:
            st.write("•",x)

        with st.expander("📘 Cara membaca Final Research Card"):
            st.write("**Fundamental** menilai kualitas bisnis, pertumbuhan, profitabilitas, utang, arus kas, dan valuasi.")
            st.write("**Ownership** melihat perubahan kepemilikan ritel dan kualitas coverage data KSEI.")
            st.write("**Technical** melihat arah harga, momentum, volume, support/resistance, dan entry setup.")
            st.write("**Market** menunjukkan apakah kondisi pasar secara umum mendukung atau justru melemah.")
            st.write("**News** hanya menggunakan berita yang lolos filter relevansi emiten.")
            st.write("Skor gabungan hanya alat ringkas. Jangan menggunakannya sebagai probabilitas naik atau rekomendasi beli/jual.")

with tab7:
    st.subheader("Portfolio Mode")
    st.write(
        "Gunakan bagian ini untuk melihat **risiko gabungan**, bukan hanya menilai saham satu per satu. "
        "Fokus utama: bobot posisi, konsentrasi sektor, korelasi, volatilitas, dan posisi yang paling perlu perhatian."
    )

    available_portfolio=sorted(set(price["Ticker"].unique()) & set(full_universe))
    selected=st.multiselect(
        "Pilih saham dalam portofolio (maksimal 12)",
        available_portfolio,
        default=[],
        max_selections=12,
        key="portfolio_tickers"
    )

    if len(selected)<2:
        st.info("Pilih minimal 2 saham agar analisis diversifikasi dan korelasi dapat dihitung.")
    else:
        allocation_mode=st.radio(
            "Metode bobot",
            ["Bobot sama","Masukkan bobot sendiri"],
            horizontal=True
        )

        weights={}
        if allocation_mode=="Bobot sama":
            for t in selected:
                weights[t]=1/len(selected)
        else:
            st.caption("Masukkan perkiraan porsi masing-masing saham. Dashboard akan menormalkan total menjadi 100%.")
            cols=st.columns(min(4,len(selected)))
            raw={}
            for i,t in enumerate(selected):
                with cols[i%len(cols)]:
                    raw[t]=st.number_input(
                        f"{t} (%)",
                        min_value=0.0,max_value=100.0,
                        value=round(100/len(selected),1),
                        step=1.0,
                        key=f"weight_{t}"
                    )
            total_raw=sum(raw.values())
            if total_raw<=0:
                st.error("Total bobot harus lebih besar dari 0.")
                weights={t:1/len(selected) for t in selected}
            else:
                weights={t:raw[t]/total_raw for t in selected}

        # Fetch sector metadata on demand from the cached fundamental engine.
        sector_map={}
        fundamental_quality={}
        with st.spinner("Menyiapkan data sektor dan kondisi posisi..."):
            for t in selected:
                try:
                    f=fetch_fundamental_snapshot(t)
                except Exception:
                    f={}
                sector=(f.get("sector") if f else None) or "Sektor belum tersedia"
                sector_map[t]=sector
                fs,flabel,_,_= _fundamental_card_score(f)
                fundamental_quality[t]=flabel

        snapshots=[]
        for t in selected:
            s=_portfolio_position_snapshot(price,own,t,mreg_raw,rconf,mreg)
            if s:
                s["Weight"]=weights.get(t,0)
                s["Sector"]=sector_map.get(t,"Sektor belum tersedia")
                s["Fundamental"]=fundamental_quality.get(t,"Data terbatas")
                priority,pscore,preason=_position_priority(s,s["Weight"])
                s["Priority"]=priority
                s["PriorityScore"]=pscore
                s["PriorityReason"]=preason
                snapshots.append(s)

        pdf=pd.DataFrame(snapshots)

        # Return matrix / portfolio stats
        retmat=_portfolio_return_matrix(price,selected,lookback=252)
        corr=retmat.corr(min_periods=40) if not retmat.empty else pd.DataFrame()
        avg_corr,max_corr,max_pair=_portfolio_corr_summary(corr)

        common=retmat.dropna(how="all").copy() if not retmat.empty else pd.DataFrame()
        weighted_ret=pd.Series(dtype=float)
        if not common.empty:
            weighted_ret=pd.Series(0.0,index=common.index)
            for t in selected:
                if t in common.columns:
                    weighted_ret=weighted_ret.add(common[t].fillna(0)*weights.get(t,0),fill_value=0)

        vol_ann=weighted_ret.std()*np.sqrt(252) if len(weighted_ret)>=40 else np.nan
        port_ret20=(1+weighted_ret.tail(20)).prod()-1 if len(weighted_ret)>=20 else np.nan
        port_ret60=(1+weighted_ret.tail(60)).prod()-1 if len(weighted_ret)>=60 else np.nan

        # Sector concentration
        sector_weights={}
        for t,w in weights.items():
            sec=sector_map.get(t,"Sektor belum tersedia")
            sector_weights[sec]=sector_weights.get(sec,0)+w
        sector_max=max(sector_weights.values()) if sector_weights else 0
        largest_sector=max(sector_weights,key=sector_weights.get) if sector_weights else "N/A"

        weak_weight=0
        if not pdf.empty:
            weak_weight=float(pdf.loc[pdf["Status"]=="Hindari dulu","Weight"].sum())

        # Header summary
        st.markdown("### Ringkasan Portofolio")
        a1,a2,a3,a4,a5=st.columns(5)
        a1.metric("Jumlah saham",len(selected))
        a2.metric("Return 20 hari","N/A" if pd.isna(port_ret20) else f"{port_ret20:+.1%}")
        a3.metric("Volatilitas tahunan","N/A" if pd.isna(vol_ann) else f"{vol_ann:.1%}")
        a4.metric("Rata-rata korelasi","N/A" if pd.isna(avg_corr) else f"{avg_corr:.2f}")
        a5.metric("Bobot evidence lemah",f"{weak_weight:.0%}")

        summary_text=_portfolio_summary_text(sector_max,avg_corr,vol_ann,weak_weight)
        if weak_weight>=.25 or sector_max>=.40 or (pd.notna(avg_corr) and avg_corr>=.60):
            st.warning(f"**Kesimpulan sederhana:** {summary_text}")
        else:
            st.info(f"**Kesimpulan sederhana:** {summary_text}")

        st.markdown("### Konsentrasi")
        c1,c2,c3=st.columns(3)
        c1.metric("Sektor terbesar",largest_sector)
        c1.caption(f"Bobot sektor: {sector_max:.0%}")
        c2.metric("Risiko volatilitas",_portfolio_risk_label(vol_ann))
        c3.metric("Korelasi tertinggi","N/A" if pd.isna(max_corr) else f"{max_corr:.2f}")
        if max_pair:
            c3.caption(f"{max_pair[0]} ↔ {max_pair[1]}")

        if sector_max>=.40:
            st.error("🔴 Lebih dari 40% portofolio terkonsentrasi pada satu sektor. Risiko sektor cukup besar.")
        elif sector_max>=.25:
            st.warning("🟡 Konsentrasi sektor mulai cukup besar. Perhatikan apakah posisi-posisi tersebut sensitif pada katalis yang sama.")
        else:
            st.success("🟢 Konsentrasi sektor relatif tersebar berdasarkan data sektor yang tersedia.")

        # Position table
        st.markdown("### Kondisi masing-masing posisi")
        if not pdf.empty:
            pdisp=pdf[[
                "Ticker","Weight","Sector","Fundamental","Status","Trend",
                "Ret20D","OwnershipText","Priority","PriorityReason"
            ]].copy()
            pdisp.columns=[
                "Kode","Bobot","Sektor","Fundamental","Evidence","Arah Harga",
                "Return 20H","Ownership","Prioritas","Alasan"
            ]
            pdisp["Bobot"]=pdisp["Bobot"].map(lambda x:f"{x:.1%}")
            pdisp["Return 20H"]=pdisp["Return 20H"].map(lambda x:"" if pd.isna(x) else f"{x:+.1%}")
            pdisp=pdisp.sort_values(
                ["Prioritas","Bobot"],
                ascending=[True,False]
            )
            st.dataframe(pdisp,use_container_width=True,hide_index=True,height=430)

        st.markdown("### Prioritas pemantauan")
        priority_df=pdf.sort_values(["PriorityScore","Weight"],ascending=[False,False]).head(5)
        for _,r in priority_df.iterrows():
            icon="🔴" if r["Priority"]=="Prioritas tinggi" else "🟡" if r["Priority"]=="Perlu dipantau" else "🟢"
            st.write(
                f"{icon} **{r['Ticker']} — {r['Priority']}** · bobot {r['Weight']:.1%} · "
                f"{r['PriorityReason']}."
            )

        # Correlation matrix
        st.markdown("### Korelasi antar-saham")
        st.write(
            "Korelasi menunjukkan seberapa sering saham bergerak searah. "
            "**Mendekati +1** = sangat searah, **mendekati 0** = hubungan lemah, "
            "**negatif** = cenderung bergerak berlawanan."
        )
        if not corr.empty:
            fig=go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                zmin=-1,zmax=1,
                text=np.round(corr.values,2),
                texttemplate="%{text}",
                hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"
            ))
            fig.update_layout(height=max(430,55*len(selected)),margin=dict(l=20,r=20,t=20,b=20))
            st.plotly_chart(fig,use_container_width=True)

            st.caption(
                f"Rata-rata korelasi: **{'N/A' if pd.isna(avg_corr) else f'{avg_corr:.2f}'}** "
                f"({_correlation_label(avg_corr)}). "
                "Korelasi historis dapat berubah saat kondisi pasar berubah."
            )

        st.markdown("### Komposisi sektor")
        sector_df=pd.DataFrame([
            {"Sektor":sec,"Bobot":w*100} for sec,w in sector_weights.items()
        ]).sort_values("Bobot",ascending=False)
        fig2=go.Figure(go.Bar(
            x=sector_df["Sektor"],
            y=sector_df["Bobot"],
            text=sector_df["Bobot"].map(lambda x:f"{x:.0f}%")
        ))
        fig2.update_layout(
            height=360,
            margin=dict(l=20,r=20,t=20,b=80),
            yaxis_title="Bobot portofolio (%)"
        )
        st.plotly_chart(fig2,use_container_width=True)

        with st.expander("📘 Cara membaca Portfolio Mode"):
            st.write("**Bobot** menunjukkan seberapa besar satu saham memengaruhi hasil portofolio.")
            st.write("**Konsentrasi sektor** tinggi berarti beberapa posisi dapat terkena risiko/katalis yang sama.")
            st.write("**Korelasi** tinggi berarti memiliki banyak saham belum tentu memberi diversifikasi yang baik jika semuanya bergerak searah.")
            st.write("**Volatilitas tahunan** adalah ukuran historis besar-kecilnya pergerakan portofolio; bukan perkiraan kerugian maksimum.")
            st.write("**Bobot evidence lemah** menunjukkan berapa bagian portofolio berada pada saham yang saat ini masuk kategori 'Hindari dulu'.")
            st.write("**Prioritas pemantauan** bukan instruksi jual. Ini hanya mengurutkan posisi yang layak diperiksa lebih dahulu.")

with tab8:
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

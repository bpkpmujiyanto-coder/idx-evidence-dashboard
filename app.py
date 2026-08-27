from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="IDX Evidence Dashboard",page_icon="📈",layout="wide")
DATA=Path(__file__).parent/"data"

# ---------------- DATA ----------------
@st.cache_data
def load_data():
    price=pd.read_csv(DATA/"price_daily.csv",parse_dates=["Date"])
    own=pd.read_csv(DATA/"ownership_long.csv",parse_dates=["Date"])
    uni=pd.read_csv(DATA/"dashboard_universe.csv")
    for c in ["Open","High","Low","Close","AdjClose","Volume"]:
        price[c]=pd.to_numeric(price[c],errors="coerce")
    for c in ["RetailPct","DeltaRetail","SnapshotPrice","Coverage"]:
        own[c]=pd.to_numeric(own[c],errors="coerce")
    price["Ticker"]=price["Ticker"].astype(str).str.upper()
    own["Ticker"]=own["Ticker"].astype(str).str.upper()
    uni["Ticker"]=uni["Ticker"].astype(str).str.upper()
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
    x["Ret5D"]=x["Close"].pct_change(5)
    x["Ret20D"]=x["Close"].pct_change(20)
    x["Ret60D"]=x["Close"].pct_change(60)
    x["DistSMA20"]=x["Close"]/x["SMA20"]-1
    x["DistSMA50"]=x["Close"]/x["SMA50"]-1
    x["High20"]=x["High"].rolling(20).max()
    x["BreakoutDistance"]=x["Close"]/x["High20"]-1

    prev_close=x["Close"].shift(1)
    tr=pd.concat([
        x["High"]-x["Low"],
        (x["High"]-prev_close).abs(),
        (x["Low"]-prev_close).abs()
    ],axis=1).max(axis=1)
    x["ATR14"]=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()

    x["Low20"]=x["Low"].rolling(20).min()
    x["Low60"]=x["Low"].rolling(60).min()
    x["High60"]=x["High"].rolling(60).max()
    return x

def trend_label(r):
    if pd.isna(r["SMA50"]): return "Data belum cukup"
    c=r["Close"]
    if pd.notna(r["SMA200"]):
        if c>r["SMA20"]>r["SMA50"]>r["SMA200"]: return "Naik kuat"
        if c<r["SMA20"]<r["SMA50"]<r["SMA200"]: return "Turun kuat"
        if c>r["SMA50"] and r["SMA50"]>r["SMA200"]: return "Cenderung naik"
        if c<r["SMA50"] and r["SMA50"]<r["SMA200"]: return "Cenderung turun"
    return "Netral / sideways"

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
        pct_above20=("A20","mean"),
        pct_above50=("A50","mean"),
        pct_above200=("A200","mean"),
        median_ret20=("R20","median")
    ).reset_index()

def regime(r,bull=.55,bear=.45):
    if pd.isna(r["median_ret20"]): return "Belum diketahui"
    if r["pct_above20"]>=bull and r["pct_above50"]>=bull and r["median_ret20"]>0:
        return "Pasar sehat"
    if r["pct_above20"]<=bear and r["pct_above50"]<=bear and r["median_ret20"]<0:
        return "Pasar lemah"
    return "Pasar netral"

def regime_raw(r,bull=.55,bear=.45):
    if pd.isna(r["median_ret20"]): return "Unknown"
    if r["pct_above20"]>=bull and r["pct_above50"]>=bull and r["median_ret20"]>0:
        return "Bull"
    if r["pct_above20"]<=bear and r["pct_above50"]<=bear and r["median_ret20"]<0:
        return "Bear"
    return "Neutral"

def regime_confidence(r):
    regs=[regime_raw(r,.50,.50),regime_raw(r,.55,.45),regime_raw(r,.60,.40)]
    conf="Tinggi" if len(set(regs))==1 else "Sedang"
    return conf,regs

def add_percentile(own):
    x=own.copy()
    x["Magnitude"]=x["DeltaRetail"].abs()
    x["OwnershipPctile"]=np.nan
    for d,g in x.groupby("Date"):
        idx=g.index[(g["DeltaRetail"]<0)&g["DeltaRetail"].notna()]
        if len(idx):
            x.loc[idx,"OwnershipPctile"]=x.loc[idx,"Magnitude"].rank(pct=True)
    return x

def ownership_text(delta,p):
    if pd.isna(delta):
        return "Belum ada perubahan"
    if delta>0:
        return "Porsi ritel meningkat"
    if pd.notna(p) and p>=.90:
        return "Ritel turun sangat besar"
    if pd.notna(p) and p>=.75:
        return "Ritel turun cukup besar"
    return "Ritel sedikit berkurang"

def volume_text(vr):
    if pd.isna(vr): return "Data volume belum cukup"
    if vr>=1.5: return "Volume sangat ramai"
    if vr>=1.0: return "Volume di atas normal"
    if vr>=.75: return "Volume normal"
    return "Volume sepi"

def momentum_text(rsi_value):
    if pd.isna(rsi_value): return "Belum tersedia"
    if rsi_value>=70: return "Sudah cukup panas"
    if rsi_value>=55: return "Momentum positif"
    if rsi_value>=45: return "Momentum netral"
    if rsi_value>=30: return "Momentum lemah"
    return "Tekanan jual tinggi"

def evidence(delta,pctile,vr,mreg_raw,conf,coverage,trend):
    score=0; pos=[]; risk=[]
    if pd.notna(delta):
        if delta<0: score+=1; pos.append("Porsi kepemilikan ritel berkurang")
        elif delta>0: score-=1; risk.append("Porsi kepemilikan ritel bertambah")

    if pd.notna(pctile):
        if pctile>=.90:
            score+=2; pos.append("Penurunan ritel termasuk 10% terbesar")
        elif pctile>=.75:
            score+=1; pos.append("Penurunan ritel termasuk 25% terbesar")

    if pd.notna(vr) and vr>1:
        score+=1; pos.append("Aktivitas transaksi lebih ramai dari biasanya")

    if pd.notna(coverage):
        if coverage>=.95:
            pos.append("Kualitas data ownership tinggi")
        elif coverage<.75:
            score-=1; risk.append("Coverage data ownership rendah")

    if trend in ("Cenderung naik","Naik kuat"):
        score+=1; pos.append("Arah harga sedang mendukung")
    elif trend in ("Cenderung turun","Turun kuat"):
        score-=1; risk.append("Arah harga masih lemah")

    if mreg_raw=="Bear":
        score-=2 if conf=="Tinggi" else 1
        risk.append("Kondisi pasar sedang lemah")
    elif mreg_raw=="Bull":
        score+=1; pos.append("Kondisi pasar sedang sehat")

    if score>=4:
        label="Menarik"
        simple="Banyak sinyal pendukung muncul bersamaan."
    elif score>=2:
        label="Perlu dipantau"
        simple="Ada beberapa sinyal positif, tetapi belum lengkap."
    elif score>=0:
        label="Netral"
        simple="Belum ada alasan kuat untuk menjadi agresif."
    else:
        label="Waspada"
        simple="Risiko atau sinyal negatif masih lebih dominan."

    return label,score,simple,pos,risk

def action_bucket(label,risk_override,trend):
    if label=="Menarik" and risk_override=="Tidak" and trend not in ("Turun kuat","Cenderung turun"):
        return "Kandidat Akumulasi"
    if label in ("Menarik","Perlu dipantau"):
        return "Watchlist"
    return "Hindari dulu"

def simple_summary(ticker,label,ownership,voltxt,trend,mkt,risk_override):
    if label=="Menarik":
        base=f"{ticker} memiliki beberapa sinyal yang saling mendukung."
    elif label=="Perlu dipantau":
        base=f"{ticker} mulai menunjukkan tanda menarik, tetapi konfirmasinya belum lengkap."
    elif label=="Netral":
        base=f"{ticker} belum menunjukkan sinyal yang cukup kuat."
    else:
        base=f"{ticker} masih memiliki lebih banyak faktor risiko daripada faktor pendukung."

    return f"{base} Ownership: {ownership}. Volume: {voltxt}. Harga: {trend}. Kondisi pasar: {mkt}."



def idx_tick_size(price):
    # IDX price fraction approximation for dashboard planning.
    if price < 200:
        return 1
    if price < 500:
        return 2
    if price < 2000:
        return 5
    if price < 5000:
        return 10
    return 25

def round_to_tick(value, mode="nearest"):
    if pd.isna(value):
        return np.nan
    tick=idx_tick_size(value)
    if mode=="down":
        return math.floor(value/tick)*tick
    if mode=="up":
        return math.ceil(value/tick)*tick
    return round(value/tick)*tick

def build_trading_plan(tf, row, market_regime):
    last=tf.iloc[-1]
    close=float(last["Close"])
    atr=float(last["ATR14"]) if pd.notna(last["ATR14"]) else close*0.03

    # Recent structural zones. Use multiple references to avoid a single arbitrary line.
    support_candidates=[
        last.get("SMA20",np.nan),
        last.get("SMA50",np.nan),
        last.get("Low20",np.nan),
    ]
    support_candidates=[float(x) for x in support_candidates if pd.notna(x) and x<close*1.03]
    if support_candidates:
        support=max([x for x in support_candidates if x<=close] or support_candidates)
    else:
        support=close-atr

    resistance_candidates=[
        last.get("High20",np.nan),
        last.get("High60",np.nan),
    ]
    resistance_candidates=[float(x) for x in resistance_candidates if pd.notna(x) and x>close]
    resistance=min(resistance_candidates) if resistance_candidates else close+2*atr

    # Entry zone depends on trend/status.
    trend=row["Trend"]
    status=row["Status"]

    if status=="Kandidat Akumulasi" and trend in ("Naik kuat","Cenderung naik"):
        entry_low=max(support,close-0.75*atr)
        entry_high=close+0.25*atr
        entry_note="Area pullback/dekat harga saat ini; hindari mengejar jika harga melonjak jauh di atas area."
    elif trend=="Netral / sideways":
        entry_low=support
        entry_high=min(close,support+0.75*atr)
        entry_note="Lebih aman menunggu harga mendekati area support atau breakout yang terkonfirmasi."
    else:
        entry_low=support
        entry_high=support+0.5*atr
        entry_note="Karena tren belum kuat, area masuk hanya untuk pemantauan; tunggu konfirmasi harga/volume."

    # Invalidation below support with ATR buffer.
    invalidation=support-0.75*atr

    # Targets: nearest resistance and extension.
    target1=max(resistance,close+1.25*atr)
    target2=max(target1+1.5*atr,close+3*atr)

    # Risk-reward calculated from mid entry.
    entry_mid=(entry_low+entry_high)/2
    risk=max(entry_mid-invalidation,0.0001)
    rr1=(target1-entry_mid)/risk
    rr2=(target2-entry_mid)/risk

    # Market regime downgrade.
    if market_regime=="Pasar lemah":
        plan_status="Tunggu"
        plan_reason="Pasar sedang lemah, jadi level teknikal sebaiknya diperlakukan lebih defensif."
    elif row["Status"]=="Hindari dulu":
        plan_status="Tunggu"
        plan_reason="Evidence saham belum cukup kuat untuk membuat rencana agresif."
    elif rr1<1.2:
        plan_status="Tunggu"
        plan_reason="Risk–reward target pertama belum menarik."
    elif row["Status"]=="Kandidat Akumulasi":
        plan_status="Layak dipantau"
        plan_reason="Evidence dan struktur harga cukup mendukung untuk membuat skenario."
    else:
        plan_status="Watchlist"
        plan_reason="Ada skenario teknikal, tetapi masih membutuhkan konfirmasi."

    vals={
        "support":round_to_tick(support),
        "resistance":round_to_tick(resistance),
        "entry_low":round_to_tick(entry_low,"down"),
        "entry_high":round_to_tick(entry_high,"up"),
        "invalidation":round_to_tick(invalidation,"down"),
        "target1":round_to_tick(target1,"up"),
        "target2":round_to_tick(target2,"up"),
        "rr1":rr1,
        "rr2":rr2,
        "atr":atr,
        "status":plan_status,
        "reason":plan_reason,
        "entry_note":entry_note,
    }
    return vals

def rupiah(v):
    if pd.isna(v):
        return "N/A"
    return f"Rp {v:,.0f}"


def analyst_card(row, market_regime):
    positives=[]
    risks=[]
    waits=[]
    invalidations=[]

    # What is interesting?
    if row["DeltaRetail"]<0:
        if pd.notna(row["OwnershipPctile"]) and row["OwnershipPctile"]>=.90:
            positives.append("Penurunan kepemilikan ritel termasuk sangat besar dibanding saham lain.")
        elif pd.notna(row["OwnershipPctile"]) and row["OwnershipPctile"]>=.75:
            positives.append("Kepemilikan ritel turun cukup besar.")
        else:
            positives.append("Kepemilikan ritel berkurang dibanding snapshot sebelumnya.")
    if pd.notna(row["VolRatio"]) and row["VolRatio"]>1:
        positives.append("Aktivitas transaksi lebih ramai dari rata-rata 20 hari.")
    if row["Trend"] in ("Naik kuat","Cenderung naik"):
        positives.append("Arah harga sedang mendukung.")
    if pd.notna(row["RSI14"]) and 50<=row["RSI14"]<70:
        positives.append("Momentum harga masih positif dan belum terlalu panas.")

    # Risks
    if row["Trend"] in ("Cenderung turun","Turun kuat"):
        risks.append("Arah harga masih lemah.")
    if pd.notna(row["RSI14"]) and row["RSI14"]>=70:
        risks.append("Momentum sudah cukup panas sehingga risiko koreksi meningkat.")
    if pd.notna(row["Coverage"]) and row["Coverage"]<.95:
        risks.append("Coverage data ownership belum penuh.")
    if row["DeltaRetail"]>0:
        risks.append("Porsi kepemilikan ritel justru meningkat.")
    if market_regime=="Pasar lemah":
        risks.append("Kondisi pasar sedang lemah dan dapat membuat sinyal saham individual gagal.")

    # What to wait for
    if row["Trend"] in ("Netral / sideways","Cenderung turun","Turun kuat"):
        waits.append("Tunggu arah harga membaik atau minimal bertahan di atas rata-rata 20/50 hari.")
    if pd.isna(row["VolRatio"]) or row["VolRatio"]<1:
        waits.append("Tunggu aktivitas transaksi meningkat di atas rata-rata.")
    if market_regime=="Pasar netral":
        waits.append("Tunggu konfirmasi dari harga/volume karena kondisi pasar belum benar-benar kuat.")
    elif market_regime=="Pasar lemah":
        waits.append("Tunggu kondisi pasar membaik sebelum menaikkan keyakinan.")
    if pd.notna(row["OwnershipPctile"]) and row["OwnershipPctile"]<.75 and row["DeltaRetail"]<0:
        waits.append("Tunggu snapshot ownership berikutnya untuk melihat apakah penurunan ritel berlanjut.")

    # Invalidation
    if pd.notna(row["SMA50"]):
        invalidations.append("Tesis teknikal melemah jika harga kembali konsisten di bawah rata-rata 50 hari.")
    if row["DeltaRetail"]<0:
        invalidations.append("Tesis ownership melemah jika snapshot berikutnya menunjukkan ritel kembali meningkat tajam.")
    if market_regime!="Pasar lemah":
        invalidations.append("Keyakinan harus diturunkan jika market regime berubah menjadi Pasar lemah.")

    if not positives:
        positives.append("Belum ada faktor positif yang cukup dominan.")
    if not risks:
        risks.append("Belum ada risiko utama dari model, tetapi risiko pasar tetap ada.")
    if not waits:
        waits.append("Tidak ada konfirmasi tambahan yang wajib; tetap pantau harga, volume, dan snapshot ownership berikutnya.")
    if not invalidations:
        invalidations.append("Tesis perlu ditinjau ulang jika arah harga dan ownership berubah berlawanan.")

    return positives,risks,waits,invalidations

def verdict_text(row, market_regime):
    if row["Status"]=="Kandidat Akumulasi":
        if market_regime=="Pasar sehat":
            return "Beberapa bukti utama mendukung dan kondisi pasar cukup kondusif."
        return "Beberapa bukti utama mendukung, tetapi tetap perlu konfirmasi karena pasar belum sepenuhnya kuat."
    if row["Status"]=="Watchlist":
        return "Ada tanda menarik, tetapi belum cukup lengkap untuk menaikkan keyakinan."
    return "Faktor risiko masih lebih dominan; lebih baik menunggu perubahan kondisi."

price,own,uni=load_data()
own=add_percentile(own)
breadth=build_breadth(price,uni)
bnow=breadth.dropna(subset=["pct_above20","pct_above50"]).iloc[-1]
mreg=regime(bnow)
mreg_raw=regime_raw(bnow)
rconf,rtests=regime_confidence(bnow)

@st.cache_data
def make_screener(price,own,uni,mreg_raw,rconf,mreg):
    rows=[]
    u100=set(uni.loc[uni["Universe100"]=="Yes","Ticker"])
    for t in sorted(set(price["Ticker"])&set(own["Ticker"])):
        p=price[price["Ticker"]==t]
        if len(p)<20:
            continue
        tf=technical_frame(p)
        last=tf.iloc[-1]
        ot=own[(own["Ticker"]==t)&own["RetailPct"].notna()].sort_values("Date")
        if ot.empty:
            continue
        lo=ot.iloc[-1]
        tr=trend_label(last)
        lab,score,simple,pos,risk=evidence(
            lo["DeltaRetail"],lo["OwnershipPctile"],last["VolRatio"],
            mreg_raw,rconf,lo["Coverage"],tr
        )
        risk_override="Ya" if any("pasar" in x.lower() and "lemah" in x.lower() for x in risk) else "Tidak"
        bucket=action_bucket(lab,risk_override,tr)
        owntext=ownership_text(lo["DeltaRetail"],lo["OwnershipPctile"])
        voltxt=volume_text(last["VolRatio"])
        rows.append({
            "Ticker":t,
            "Status":bucket,
            "Kesimpulan":lab,
            "Ringkasan":simple_summary(t,lab,owntext,voltxt,tr,mreg,risk_override),
            "EvidenceScore":score,
            "Close":last["Close"],
            "Ret5D":last["Ret5D"],
            "Ret20D":last["Ret20D"],
            "Ret60D":last["Ret60D"],
            "RetailPct":lo["RetailPct"],
            "DeltaRetail":lo["DeltaRetail"],
            "OwnershipPctile":lo["OwnershipPctile"],
            "OwnershipText":owntext,
            "Coverage":lo["Coverage"],
            "VolRatio":last["VolRatio"],
            "VolumeText":voltxt,
            "RSI14":last["RSI14"],
            "MomentumText":momentum_text(last["RSI14"]),
            "Trend":tr,
            "SMA20":last["SMA20"],
            "SMA50":last["SMA50"],
            "SMA200":last["SMA200"],
            "DistSMA20":last["DistSMA20"],
            "DistSMA50":last["DistSMA50"],
            "BreakoutDistance":last["BreakoutDistance"],
            "ATR14":last["ATR14"],
            "Low20":last["Low20"],
            "Low60":last["Low60"],
            "High20":last["High20"],
            "High60":last["High60"],
            "RiskOverride":risk_override,
            "Universe100":"Ya" if t in u100 else "Tidak",
        })
    return pd.DataFrame(rows)

scr=make_screener(price,own,uni,mreg_raw,rconf,mreg)

# ---------------- UI ----------------
st.title("IDX Evidence Dashboard")
st.caption("Versi sederhana: membantu memahami kondisi saham tanpa harus menguasai istilah teknikal.")

# Market status in plain language
c1,c2,c3,c4=st.columns(4)
c1.metric("Kondisi pasar",mreg,f"Keyakinan {rconf.lower()}")
c2.metric("Saham di atas MA20",f"{bnow['pct_above20']:.0%}")
c3.metric("Saham di atas MA50",f"{bnow['pct_above50']:.0%}")
c4.metric("Median return 20 hari",f"{bnow['median_ret20']:.1%}")

if mreg=="Pasar sehat":
    st.success("🟢 Kondisi pasar cukup mendukung. Sinyal saham individual cenderung lebih layak diperhatikan.")
elif mreg=="Pasar lemah":
    st.error("🔴 Pasar sedang lemah. Sinyal positif saham individual perlu diperlakukan lebih hati-hati.")
else:
    st.info("🟡 Pasar sedang netral. Pilih saham secara selektif dan tunggu konfirmasi.")

tab1,tab2,tab3,tab4=st.tabs([
    "⭐ Pilihan Hari Ini",
    "🔎 Semua Saham",
    "📊 Detail Saham",
    "🌐 Kondisi Pasar"
])

# ---------- Easy ranking ----------
with tab1:
    st.subheader("Pilihan Hari Ini")
    st.write(
        "Bagian ini menyaring saham yang paling layak **diteliti lebih lanjut**, "
        "bukan daftar rekomendasi beli."
    )

    st.caption(
        "Urutan membaca: lihat status → baca ringkasan → buka Detail Saham untuk melihat "
        "apa yang menarik, risikonya, konfirmasi yang perlu ditunggu, dan kondisi pembatalan tesis."
    )

    focus=scr.sort_values(
        ["EvidenceScore","OwnershipPctile","VolRatio"],
        ascending=[False,False,False]
    )

    a,b,c=st.columns(3)
    a.metric("Kandidat Akumulasi",int((focus["Status"]=="Kandidat Akumulasi").sum()))
    b.metric("Watchlist",int((focus["Status"]=="Watchlist").sum()))
    c.metric("Hindari dulu",int((focus["Status"]=="Hindari dulu").sum()))

    show=focus[focus["Status"].isin(["Kandidat Akumulasi","Watchlist"])].head(12)

    if show.empty:
        st.warning("Belum ada saham yang memenuhi filter menarik saat ini.")
    else:
        for _,r in show.iterrows():
            if r["Status"]=="Kandidat Akumulasi":
                icon="🟢"
            else:
                icon="🟡"

            with st.container(border=True):
                x1,x2,x3,x4=st.columns([1,1.3,1.2,3.5])
                x1.markdown(f"### {icon} {r['Ticker']}")
                x1.write(f"**{r['Status']}**")
                x2.metric("Harga",f"Rp {r['Close']:,.0f}")
                x2.caption(f"20 hari: {r['Ret20D']:+.1%}" if pd.notna(r["Ret20D"]) else "20 hari: N/A")
                x3.write(f"**Ownership**\n\n{r['OwnershipText']}")
                x3.write(f"**Volume**\n\n{r['VolumeText']}")
                x4.write(r["Ringkasan"])

# ---------- Full screener ----------
with tab2:
    st.subheader("Semua Saham")
    st.caption("Gunakan filter sederhana untuk mempersempit saham yang ingin diperiksa.")

    f1,f2,f3,f4=st.columns(4)
    status_filter=f1.multiselect(
        "Status",
        ["Kandidat Akumulasi","Watchlist","Hindari dulu"],
        default=["Kandidat Akumulasi","Watchlist"]
    )
    own_filter=f2.selectbox(
        "Kondisi ownership",
        ["Semua","Ritel turun sangat besar","Ritel turun cukup besar","Ritel sedikit berkurang","Porsi ritel meningkat"]
    )
    trend_filter=f3.selectbox(
        "Arah harga",
        ["Semua","Naik kuat","Cenderung naik","Netral / sideways","Cenderung turun","Turun kuat"]
    )
    min_cov=f4.slider("Minimal kualitas data",0,100,75,5)/100

    v=scr.copy()
    if status_filter:
        v=v[v["Status"].isin(status_filter)]
    if own_filter!="Semua":
        v=v[v["OwnershipText"]==own_filter]
    if trend_filter!="Semua":
        v=v[v["Trend"]==trend_filter]
    v=v[(v["Coverage"].isna())|(v["Coverage"]>=min_cov)]

    v=v.sort_values(["EvidenceScore","OwnershipPctile","VolRatio"],ascending=[False,False,False])

    display=v[[
        "Ticker","Status","Kesimpulan","Close","Ret20D",
        "OwnershipText","VolumeText","Trend","MomentumText","RiskOverride"
    ]].copy()
    display.columns=[
        "Kode","Status","Kesimpulan","Harga","Return 20H",
        "Ownership","Volume","Arah Harga","Momentum","Peringatan Pasar"
    ]
    display["Harga"]=display["Harga"].map(lambda x:f"Rp {x:,.0f}")
    display["Return 20H"]=display["Return 20H"].map(lambda x:"" if pd.isna(x) else f"{x:+.1%}")
    st.dataframe(display,use_container_width=True,hide_index=True,height=560)

    with st.expander("Lihat data teknis lanjutan"):
        tech=v[[
            "Ticker","EvidenceScore","RetailPct","DeltaRetail","OwnershipPctile",
            "Coverage","VolRatio","RSI14","Ret5D","Ret20D","Ret60D",
            "DistSMA20","DistSMA50","BreakoutDistance"
        ]].copy()
        st.dataframe(tech,use_container_width=True,hide_index=True)

# ---------- Detail ----------
with tab3:
    ticker=st.selectbox("Pilih saham",sorted(scr["Ticker"].unique()))
    row=scr[scr["Ticker"]==ticker].iloc[0]
    p=price[price["Ticker"]==ticker]
    tf=technical_frame(p)
    ot=own[(own["Ticker"]==ticker)&own["RetailPct"].notna()].sort_values("Date")

    st.subheader(f"{ticker} — {row['Status']}")
    if row["Status"]=="Kandidat Akumulasi":
        st.success(row["Ringkasan"])
    elif row["Status"]=="Watchlist":
        st.warning(row["Ringkasan"])
    else:
        st.error(row["Ringkasan"])

    m1,m2,m3,m4,m5=st.columns(5)
    m1.metric("Harga",f"Rp {row['Close']:,.0f}")
    m2.metric("20 hari","N/A" if pd.isna(row["Ret20D"]) else f"{row['Ret20D']:+.1%}")
    m3.metric("Ownership",row["OwnershipText"])
    m4.metric("Volume",row["VolumeText"])
    m5.metric("Arah harga",row["Trend"])

    st.markdown("### Cara membacanya")
    c1,c2,c3,c4=st.columns(4)
    c1.info(f"**1. Ownership**\n\n{row['OwnershipText']}")
    c2.info(f"**2. Aktivitas transaksi**\n\n{row['VolumeText']}")
    c3.info(f"**3. Arah harga**\n\n{row['Trend']}")
    c4.info(f"**4. Momentum**\n\n{row['MomentumText']}")


    st.markdown("### Kartu Analisis Saham")
    positives,risks,waits,invalidations=analyst_card(row,mreg)

    if row["Status"]=="Kandidat Akumulasi":
        st.success(f"**Kesimpulan:** {verdict_text(row,mreg)}")
    elif row["Status"]=="Watchlist":
        st.warning(f"**Kesimpulan:** {verdict_text(row,mreg)}")
    else:
        st.error(f"**Kesimpulan:** {verdict_text(row,mreg)}")

    q1,q2=st.columns(2)
    with q1:
        with st.container(border=True):
            st.markdown("#### ✅ Apa yang menarik?")
            for x in positives:
                st.write("•",x)
        with st.container(border=True):
            st.markdown("#### ⏳ Apa yang perlu ditunggu?")
            for x in waits:
                st.write("•",x)
    with q2:
        with st.container(border=True):
            st.markdown("#### ⚠️ Apa risikonya?")
            for x in risks:
                st.write("•",x)
        with st.container(border=True):
            st.markdown("#### ❌ Kapan tesis dianggap batal?")
            for x in invalidations:
                st.write("•",x)


    st.markdown("### Trading Plan Sederhana")
    plan=build_trading_plan(tf,row,mreg)

    if plan["status"]=="Layak dipantau":
        st.success(f"**{plan['status']}** — {plan['reason']}")
    elif plan["status"]=="Watchlist":
        st.warning(f"**{plan['status']}** — {plan['reason']}")
    else:
        st.info(f"**{plan['status']}** — {plan['reason']}")

    t1,t2,t3,t4,t5=st.columns(5)
    t1.metric("Area support",rupiah(plan["support"]))
    t2.metric("Area masuk",f"{rupiah(plan['entry_low'])} – {rupiah(plan['entry_high'])}")
    t3.metric("Batas batal",rupiah(plan["invalidation"]))
    t4.metric("Target 1",rupiah(plan["target1"]),f"RR {plan['rr1']:.1f}x")
    t5.metric("Target 2",rupiah(plan["target2"]),f"RR {plan['rr2']:.1f}x")

    st.caption(plan["entry_note"])

    with st.expander("Cara membaca Trading Plan"):
        st.write("**Support** = area harga yang secara teknikal berpotensi menahan penurunan, bukan jaminan harga akan memantul.")
        st.write("**Area masuk** = rentang harga untuk skenario, bukan harga beli wajib.")
        st.write("**Batas batal** = jika harga turun melewati area ini, skenario teknikal perlu dievaluasi ulang.")
        st.write("**Target 1 & 2** = area resistance/ekstensi berbasis struktur harga dan volatilitas.")
        st.write("**RR (risk–reward)** = perbandingan potensi kenaikan terhadap risiko menuju batas batal. Semakin besar umumnya semakin menarik, tetapi probabilitas tetap tidak pasti.")

    left,right=st.columns([1.65,1])
    with left:
        st.markdown("#### Grafik harga")
        fig=go.Figure()
        fig.add_trace(go.Candlestick(
            x=tf["Date"],open=tf["Open"],high=tf["High"],
            low=tf["Low"],close=tf["Close"],name="Harga"
        ))
        fig.add_trace(go.Scatter(x=tf["Date"],y=tf["SMA20"],name="Rata-rata 20 hari"))
        fig.add_trace(go.Scatter(x=tf["Date"],y=tf["SMA50"],name="Rata-rata 50 hari"))
        fig.add_trace(go.Scatter(x=tf["Date"],y=tf["SMA200"],name="Rata-rata 200 hari"))
        fig.add_hline(y=plan["support"],line_dash="dot",annotation_text="Support")
        fig.add_hline(y=plan["invalidation"],line_dash="dash",annotation_text="Batas batal")
        fig.add_hline(y=plan["target1"],line_dash="dot",annotation_text="Target 1")
        fig.add_hline(y=plan["target2"],line_dash="dot",annotation_text="Target 2")
        fig.update_layout(height=500,xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig,use_container_width=True)

    with right:
        st.markdown("#### Kepemilikan ritel")
        fig2=go.Figure(go.Scatter(
            x=ot["Date"],y=ot["RetailPct"]*100,
            mode="lines+markers",name="Retail"
        ))
        fig2.update_layout(height=260,margin=dict(l=10,r=10,t=20,b=10),yaxis_title="% Retail")
        st.plotly_chart(fig2,use_container_width=True)

        st.markdown("#### Perubahan tiap snapshot")
        fig3=go.Figure(go.Bar(x=ot["Date"],y=ot["DeltaRetail"]*100,name="Perubahan"))
        fig3.update_layout(height=220,margin=dict(l=10,r=10,t=20,b=10),yaxis_title="Perubahan (%)")
        st.plotly_chart(fig3,use_container_width=True)

    with st.expander("Penjelasan istilah"):
        st.write("**Ownership turun**: porsi saham yang dikategorikan sebagai kepemilikan ritel berkurang dibanding snapshot sebelumnya.")
        st.write("**Volume di atas normal**: transaksi hari terakhir lebih ramai dibanding rata-rata 20 hari.")
        st.write("**Arah harga**: ringkasan posisi harga terhadap rata-rata 20, 50, dan 200 hari.")
        st.write("**Momentum**: gambaran kekuatan kenaikan/penurunan harga menggunakan RSI.")
        st.write("**Kandidat Akumulasi** tidak berarti pasti naik. Label hanya berarti beberapa bukti pendukung muncul bersamaan.")

# ---------- Market ----------
with tab4:
    st.subheader("Kondisi Pasar")
    st.write(
        "Bagian ini menjawab pertanyaan sederhana: "
        "**apakah sebagian besar saham sedang sehat atau justru melemah?**"
    )

    b=breadth.tail(220)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=b["Date"],y=b["pct_above20"]*100,name="% saham di atas rata-rata 20 hari"))
    fig.add_trace(go.Scatter(x=b["Date"],y=b["pct_above50"]*100,name="% saham di atas rata-rata 50 hari"))
    fig.add_hline(y=55,line_dash="dash",annotation_text="Area pasar sehat")
    fig.add_hline(y=45,line_dash="dash",annotation_text="Area pasar lemah")
    fig.update_layout(height=430,margin=dict(l=10,r=10,t=20,b=10),yaxis_title="% saham")
    st.plotly_chart(fig,use_container_width=True)

    st.markdown("#### Kesimpulan sederhana")
    if mreg=="Pasar sehat":
        st.success("Lebih banyak saham berada dalam kondisi teknikal sehat. Risiko pasar secara umum lebih rendah.")
    elif mreg=="Pasar lemah":
        st.error("Banyak saham sedang melemah. Sinyal saham individual lebih mudah gagal.")
    else:
        st.info("Pasar belum menunjukkan arah yang benar-benar kuat. Seleksi saham perlu lebih ketat.")

st.divider()
d1,d2=st.columns(2)
d1.metric("Harga terakhir dalam data",price["Date"].max().strftime("%d %b %Y"))
d2.metric("Snapshot ownership terakhir",own["Date"].max().strftime("%d %b %Y"))

st.caption(
    "Label di dashboard adalah alat bantu riset, bukan rekomendasi investasi. "
    "Kepemilikan broker/ritel tidak identik dengan satu pihak dan indikator teknikal bersifat probabilistik."
)

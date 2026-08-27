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
            "DistSMA20":last["DistSMA20"],
            "DistSMA50":last["DistSMA50"],
            "BreakoutDistance":last["BreakoutDistance"],
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

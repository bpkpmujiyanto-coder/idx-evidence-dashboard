from pathlib import Path
import argparse, time
import pandas as pd
import yfinance as yf

BASE=Path(__file__).resolve().parent
DATA=BASE/"data"
PRICE=DATA/"price_daily.csv"
UNIVERSE=DATA/"dashboard_universe.csv"
STATUS=DATA/"bootstrap_status.csv"

def fetch_one(ticker,start):
    x=yf.download(
        f"{ticker}.JK",start=start,interval="1d",auto_adjust=False,actions=False,
        progress=False,multi_level_index=False,threads=False
    )
    if x is None or x.empty:return pd.DataFrame()
    x=x.reset_index().rename(columns={"Adj Close":"AdjClose"})
    x["Ticker"]=ticker
    x["SourceType"]="Yahoo Finance via yfinance"
    cols=["Date","Ticker","Open","High","Low","Close","AdjClose","Volume","SourceType"]
    for c in cols:
        if c not in x.columns:x[c]=pd.NA
    return x[cols]

def save_combined(old,newframes):
    if newframes:
        new=pd.concat(newframes,ignore_index=True)
        combined=pd.concat([old,new],ignore_index=True)
    else:
        combined=old.copy()
    combined["Date"]=pd.to_datetime(combined["Date"])
    combined["Ticker"]=combined["Ticker"].astype(str).str.upper()
    combined=(combined.sort_values(["Ticker","Date"])
              .drop_duplicates(["Ticker","Date"],keep="last")
              .sort_values(["Ticker","Date"]))
    combined["Date"]=combined["Date"].dt.date
    combined.to_csv(PRICE,index=False)
    return pd.read_csv(PRICE,parse_dates=["Date"])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",default="2025-03-27")
    ap.add_argument("--sleep",type=float,default=.35)
    ap.add_argument("--checkpoint",type=int,default=25)
    args=ap.parse_args()

    old=pd.read_csv(PRICE,parse_dates=["Date"])
    old["Ticker"]=old["Ticker"].astype(str).str.upper()
    universe=pd.read_csv(UNIVERSE)["Ticker"].dropna().astype(str).str.upper().unique().tolist()

    counts=old.groupby("Ticker").size().to_dict()
    todo=[t for t in universe if counts.get(t,0)<120]
    print(f"Universe: {len(universe)} | Need bootstrap: {len(todo)}")

    status=[]; frames=[]
    for i,t in enumerate(todo,1):
        try:
            x=fetch_one(t,args.start)
            if x.empty:
                status.append([t,"NO_DATA",0,""])
                print(f"[{i}/{len(todo)}] {t}: NO_DATA")
            else:
                frames.append(x); status.append([t,"OK",len(x),""])
                print(f"[{i}/{len(todo)}] {t}: {len(x)} rows")
        except Exception as e:
            status.append([t,"ERROR",0,str(e)[:250]])
            print(f"[{i}/{len(todo)}] {t}: ERROR {e}")
        if i%args.checkpoint==0:
            old=save_combined(old,frames); frames=[]
            pd.DataFrame(status,columns=["Ticker","Status","Rows","Message"]).to_csv(STATUS,index=False)
            print("Checkpoint saved.")
        time.sleep(args.sleep)

    old=save_combined(old,frames)
    pd.DataFrame(status,columns=["Ticker","Status","Rows","Message"]).to_csv(STATUS,index=False)
    print(f"Done. Tickers with prices: {old['Ticker'].nunique()}")

if __name__=="__main__":
    main()

from __future__ import annotations
from pathlib import Path
from datetime import timedelta
import argparse, time
import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
PRICE_FILE = DATA / "price_daily.csv"
UNIVERSE_FILE = DATA / "dashboard_universe.csv"

def load_universe():
    u = pd.read_csv(UNIVERSE_FILE)
    return sorted(u["Ticker"].dropna().astype(str).str.upper().unique().tolist())

def fetch_one(ticker, start, end=None):
    yt = f"{ticker}.JK"
    df = yf.download(
        yt, start=start, end=end, interval="1d",
        auto_adjust=False, actions=False, progress=False,
        multi_level_index=False, threads=False
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index().rename(columns={"Adj Close":"AdjClose"})
    df["Ticker"] = ticker
    df["SourceType"] = "Yahoo Finance via yfinance"
    wanted = ["Date","Ticker","Open","High","Low","Close","AdjClose","Volume","SourceType"]
    for c in wanted:
        if c not in df.columns:
            df[c] = pd.NA
    return df[wanted]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback-days", type=int, default=10)
    ap.add_argument("--sleep", type=float, default=0.15)
    args = ap.parse_args()

    old = pd.read_csv(PRICE_FILE, parse_dates=["Date"])
    old["Ticker"] = old["Ticker"].astype(str).str.upper()
    latest = old["Date"].max()
    start = (latest - timedelta(days=args.lookback_days)).date().isoformat()

    frames, status = [], []
    tickers = load_universe()
    print(f"Refreshing {len(tickers)} tickers from {start} ...")

    for i,t in enumerate(tickers,1):
        try:
            x = fetch_one(t, start)
            if x.empty:
                status.append([t,"NO_DATA",0,""])
                print(f"[{i}/{len(tickers)}] {t}: NO_DATA")
            else:
                x["Date"] = pd.to_datetime(x["Date"])
                frames.append(x)
                status.append([t,"OK",len(x),""])
                print(f"[{i}/{len(tickers)}] {t}: {len(x)} rows")
        except Exception as e:
            status.append([t,"ERROR",0,str(e)[:250]])
            print(f"[{i}/{len(tickers)}] {t}: ERROR {e}")
        time.sleep(args.sleep)

    pd.DataFrame(status, columns=["Ticker","Status","Rows","Message"]).to_csv(DATA/"update_status.csv", index=False)

    if not frames:
        raise SystemExit("No new data fetched. Existing dataset was not changed.")

    new = pd.concat(frames, ignore_index=True)
    combined = pd.concat([old, new], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined["Ticker"] = combined["Ticker"].astype(str).str.upper()
    combined = combined.sort_values(["Ticker","Date"]).drop_duplicates(["Ticker","Date"], keep="last")
    combined["Date"] = combined["Date"].dt.date
    combined.to_csv(PRICE_FILE, index=False)

    print(f"Update complete. Rows: {len(combined):,}. Latest date: {combined['Date'].max()}")

if __name__ == "__main__":
    main()

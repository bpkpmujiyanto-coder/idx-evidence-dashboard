# IDX Evidence Dashboard

Dashboard riset saham Indonesia yang menggabungkan:

- KSEI ownership / Retail %
- magnitude perubahan ownership
- volume confirmation
- SMA20 / SMA50 / SMA200
- RSI / MACD
- market breadth
- Bull / Neutral / Bear regime
- regime confidence
- Evidence Level

Evidence Level adalah alat riset, bukan rekomendasi investasi.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Update prices

```bash
python update_prices.py --lookback-days 10
```

## Data architecture

`data/price_daily.csv`  
Harga harian untuk dashboard.

`data/ownership_long.csv`  
Snapshot kepemilikan KSEI.

`data/dashboard_universe.csv`  
Universe saham dashboard.

## Automatic update

GitHub Actions workflow:
`.github/workflows/update_prices.yml`

Default schedule:
Senin–Jumat pukul 18:30 WIB.

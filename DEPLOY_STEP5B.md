# Step 5B — Deploy Ready

## Jalankan lokal
Pertama kali:
`python -m pip install -r requirements.txt`

Sesudah itu cukup double-click:
`UPDATE_AND_RUN.bat`

## Deploy gratis ke Streamlit Community Cloud
1. Buat repository GitHub baru.
2. Upload seluruh isi folder ini, termasuk folder `.github`.
3. Buka Streamlit Community Cloud dan login dengan GitHub.
4. Create app → pilih repository → main file `app.py` → Deploy.

## Update harga otomatis
Workflow `.github/workflows/update_prices.yml` berjalan Senin–Jumat pukul 18:30 WIB.
Ia menjalankan `update_prices.py`, memperbarui `data/price_daily.csv`, lalu commit hasilnya.

Manual:
GitHub → Actions → Update IDX prices → Run workflow.

## Catatan
- Harga: Yahoo Finance/yfinance sebagai ingestion sekunder.
- IDX/KSEI tetap referensi validasi.
- Ownership tidak diestimasi antar-snapshot.
- Evidence Level bukan rekomendasi beli/jual.

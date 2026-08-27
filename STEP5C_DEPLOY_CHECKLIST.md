# STEP 5C — GitHub + Streamlit Deployment Checklist

## 1. Buat repository GitHub
Nama yang disarankan:
`idx-evidence-dashboard`

Pilih:
- Public, jika dashboard boleh diakses publik
- Private, jika hanya untuk penggunaan terbatas

Jangan centang "Add README" jika Anda akan upload seluruh isi ZIP ini.

## 2. Upload file
Upload SELURUH isi folder ini ke root repository.

Pastikan root repository berisi:
- app.py
- requirements.txt
- update_prices.py
- data/
- .github/workflows/update_prices.yml
- .streamlit/config.toml

## 3. Deploy ke Streamlit Community Cloud
Masuk ke Streamlit Community Cloud menggunakan GitHub.

Pilih:
Create app → Yup, I have an app

Isi:
- Repository: <username>/idx-evidence-dashboard
- Branch: main
- Main file path: app.py

Klik Deploy.

## 4. Setelah aplikasi tampil
Cek:
- dropdown ticker bekerja;
- candlestick tampil;
- ownership chart tampil;
- Market Regime tampil;
- Evidence Level tampil;
- bagian data freshness tampil.

## 5. Aktifkan update otomatis
Di GitHub:
Actions → Update IDX prices → Run workflow

Jika berhasil, workflow akan memperbarui:
`data/price_daily.csv`

Setelah itu GitHub Actions akan berjalan otomatis Senin–Jumat 18:30 WIB.

## 6. Jika workflow tidak bisa push
GitHub:
Settings → Actions → General → Workflow permissions

Pilih:
Read and write permissions

Kemudian jalankan ulang workflow.

## 7. Catatan data
Harga: Yahoo Finance/yfinance sebagai ingestion sekunder.
Ownership: hanya berubah saat ada snapshot KSEI baru.
Evidence Level bukan rekomendasi beli/jual.

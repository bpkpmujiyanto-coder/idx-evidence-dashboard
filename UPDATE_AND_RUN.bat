@echo off
title IDX Evidence Dashboard
python update_prices.py --lookback-days 10
python -m streamlit run app.py
pause

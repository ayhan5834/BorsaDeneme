# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:42:45 2026

@author: EmirAysu
"""


import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.linear_model import HuberRegressor
import streamlit as st

st.set_page_config(page_title="Mobil Borsa", layout="centered")

# Veritabanı
conn = sqlite3.connect("takip_listesi.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS watchlist (hisse_kodu TEXT PRIMARY KEY, maliyet REAL, adet INTEGER)")
conn.commit()

st.title("📱 Mobil Borsa")
sekme1, sekme2, sekme3 = st.tabs(["PORTFÖY", "ANALİZ", "RADAR"])

with sekme1:
    st.subheader("💼 Portföy Ekle")
    # FORM YOK, DOĞRUDAN GİRDİLER
    kod = st.text_input("Hisse Kodu (Örn: THYAO)", key="h_kod").upper().strip()
    mal = st.number_input("Maliyet", value=0.0, key="h_mal")
    adet = st.number_input("Adet", value=0, step=1, key="h_adet")
    
    if st.button("SİSTEME KAYDET"):
        if kod:
            cursor.execute("INSERT OR REPLACE INTO watchlist VALUES (?, ?, ?)", (kod, mal, adet))
            conn.commit()
            st.success(f"{kod} kaydedildi!")
            st.rerun()

    st.divider()
    for satir in cursor.execute("SELECT * FROM watchlist").fetchall():
        c1, c2 = st.columns([3, 1])
        c1.write(f"**{satir[0]}** | Adet: {satir[2]} | Mal: {satir[1]}")
        if c2.button("Sil", key=f"del_{satir[0]}"):
            cursor.execute("DELETE FROM watchlist WHERE hisse_kodu = ?", (satir[0],))
            conn.commit()
            st.rerun()

with sekme2:
    st.subheader("🔍 Hızlı Analiz")
    kod_a = st.text_input("Hisse:", key="a_kod").upper().strip()
    if kod_a and st.button("Analiz Et"):
        df = yf.download(f"{kod_a}.IS", period="60d", progress=False)
        if not df.empty:
            son = df['Close'].iloc[-1].item()
            st.metric("Fiyat", f"{son:.2f} TL")
            if st.button("Listeye Ekle"):
                cursor.execute("INSERT OR REPLACE INTO watchlist VALUES (?, ?, ?)", (kod_a, 0.0, 0))
                conn.commit()
                st.rerun()

with sekme3:
    st.subheader("🚀 Mega Radar")
    if st.button("TARAMAYI BAŞLAT"):
        liste = ["THYAO", "ASELS", "AKBNK", "BIMAS"]
        for h in liste:
            try:
                df = yf.download(f"{h}.IS", period="30d", progress=False)
                rsi = ta.momentum.rsi(df['Close'].squeeze(), window=14).iloc[-1]
                if rsi < 35: st.write(f"✅ {h} (RSI: {rsi:.1f})")
            except: continue

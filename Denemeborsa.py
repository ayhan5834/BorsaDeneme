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

# --- 1. AYARLAR ---
st.set_page_config(page_title="Mobil Borsa", layout="centered")
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #FFFFFF; }
    div[data-testid="stMetricWidget"] { background-color: #1E1E1E; border: 1px solid #2D2D2D; padding: 10px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. VERİTABANI BAĞLANTISI ---
def get_db():
    conn = sqlite3.connect("takip_listesi.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            hisse_kodu TEXT PRIMARY KEY,
            maliyet REAL,
            adet INTEGER
        )
    """)
    conn.commit()
    return conn, cursor

conn, cursor = get_db()

# --- 3. ANALİZ MOTORU ---
def tahmin_et(df):
    if df is None or df.empty or len(df) < 10:
        return 0.0
    data = df.tail(60).copy()
    data['gun'] = range(len(data))
    model = HuberRegressor()
    model.fit(data[['gun']], data['Close'].squeeze())
    return model.predict([[len(data) + 4]])[0]

# --- 4. ARAYÜZ (SEKMELER) ---
st.title("📱 Mobil Borsa")
sekme1, sekme2, sekme3 = st.tabs(["PORTFÖY", "ANALİZ", "RADAR"])

with sekme1:
    st.subheader("💼 Portföyüm")
    with st.form("ekle_form", clear_on_submit=True):
        kod = st.text_input("Hisse Kodu (Örn: THYAO)").upper().strip()
        mal = st.number_input("Maliyet", value=0.0)
        adet = st.number_input("Adet", value=0, step=1)
        if st.form_submit_button("EKLE / GÜNCELLE"):
            if kod:
                cursor.execute("INSERT OR REPLACE INTO watchlist VALUES (?, ?, ?)", (kod, mal, adet))
                conn.commit()
                st.rerun()

    hisseler = cursor.execute("SELECT * FROM watchlist").fetchall()
    for h, m, a in hisseler:
        c1, c2 = st.columns([3, 1])
        c1.write(f"**{h}** | {a} Adet | {m} TL")
        if c2.button("Sil", key=f"del_{h}"):
            cursor.execute("DELETE FROM watchlist WHERE hisse_kodu = ?", (h,))
            conn.commit()
            st.rerun()

with sekme2:
    st.subheader("🔍 Hızlı Analiz")
    kod_analiz = st.text_input("Hisse Kodu:", key="analiz_inp").upper().strip()
    if kod_analiz:
        df = yf.download(f"{kod_analiz}.IS", period="60d", progress=False)
        if not df.empty:
            son = df['Close'].iloc[-1].item()
            hedef = tahmin_et(df)
            st.metric(f"{kod_analiz} Fiyat", f"{son:.2f} TL")
            st.write(f"Tahmin: {hedef:.2f} TL")
            if st.button("Portföye Ekle"):
                cursor.execute("INSERT OR REPLACE INTO watchlist VALUES (?, ?, ?)", (kod_analiz, 0.0, 0))
                conn.commit()
                st.success("Eklendi!")

with sekme3:
    st.subheader("🚀 Mega Radar")
    if st.button("TARAMAYI BAŞLAT"):
        liste = ["THYAO", "ASELS", "AKBNK", "BIMAS", "ALARK", "EREGL"]
        bulunanlar = []
        for h in liste:
            try:
                df = yf.download(f"{h}.IS", period="30d", progress=False)
                if not df.empty:
                    rsi = ta.momentum.rsi(df['Close'].squeeze(), window=14).iloc[-1]
                    if rsi < 35: bulunanlar.append(h)
            except: continue
        st.write("Sinyaller:", bulunanlar if bulunanlar else "Bulunamadı.")

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

# Streamlit yapılandırması
st.set_page_config(page_title="Mobil Borsa", layout="centered")

# CSS Stilleri
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #FFFFFF; }
    div[data-testid="stExpander"] { background-color: #1E1E1E; border: 1px solid #2D2D2D; border-radius: 10px; }
    .stButton>button { width: 100%; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# VERİTABANI YÖNETİMİ
# ==============================================================================
def veritabani_baglan():
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

conn, cursor = veritabani_baglan()

# ==============================================================================
# HESAPLAMA MOTORLARI
# ==============================================================================
def mobil_tahmin_motoru(df):
    if df is None or df.empty or len(df) < 5:
        return 0.0, np.zeros(5)
    try:
        data = df.tail(60).copy()
        data['gun'] = range(len(data))
        model = HuberRegressor(max_iter=1000)
        model.fit(data[['gun']], data['Close'].squeeze())
        gelecek = np.arange(len(data), len(data) + 5).reshape(-1, 1)
        tahmin = model.predict(gelecek)
        return tahmin[-1], tahmin
    except:
        return df['Close'].iloc[-1], np.full(5, df['Close'].iloc[-1])

# ==============================================================================
# STREAMLIT ARAYÜZÜ
# ==============================================================================
st.title("📱 Mobil Borsa Paneli")
sekme1, sekme2, sekme3 = st.tabs(["PORTFÖY", "ANALİZ", "RADAR"])

# --- 1. SEKME: PORTFÖY ---
with sekme1:
    st.subheader("💼 Portföy Yönetimi")
    
    # Form kullanımı veri girişini mobilde garantiler
    with st.form("hisse_ekle_form", clear_on_submit=True):
        yeni_kod = st.text_input("Hisse Kodu (örn: THYAO)").upper().strip()
        maliyet = st.number_input("Maliyet", value=0.0, step=0.1)
        adet = st.number_input("Adet", value=0, step=1)
        submit = st.form_submit_button("EKLE / GÜNCELLE")
        
        if submit and yeni_kod:
            cursor.execute("INSERT OR REPLACE INTO watchlist VALUES (?, ?, ?)", (yeni_kod, maliyet, adet))
            conn.commit()
            st.success(f"{yeni_kod} listeye eklendi!")
            st.rerun()

    st.divider()
    hisseler = cursor.execute("SELECT * FROM watchlist").fetchall()
    for h, m, a in hisseler:
        col1, col2 = st.columns([3, 1])
        col1.write(f"**{h}** | Maliyet: {m} | Adet: {a}")
        if col2.button("Sil", key=f"del_{h}"):
            cursor.execute("DELETE FROM watchlist WHERE hisse_kodu = ?", (h,))
            conn.commit()
            st.rerun()

# --- 2. SEKME: ANALİZ ---
with sekme2:
    st.subheader("🔍 Hızlı Hisse Analiz")
    kod = st.text_input("Hisse Analiz:", key="analiz_input").upper().strip()
    if kod:
        df = yf.download(f"{kod}.IS", period="60d", progress=False)
        if not df.empty:
            son = df['Close'].iloc[-1].item()
            hedef, _ = mobil_tahmin_motoru(df)
            st.metric(label=f"{kod} Fiyat", value=f"{son:.2f} TL")
            st.write(f"YZ 5 Günlük Tahmin: {hedef:.2f} TL")
            if st.button("Portföye Ekle"):
                cursor.execute("INSERT OR REPLACE INTO watchlist VALUES (?, ?, ?)", (kod, 0.0, 0))
                conn.commit()
                st.success("Eklendi!")

# --- 3. SEKME: RADAR ---
with sekme3:
    st.subheader("🚀 Mega Radar")
    if st.button("TARAMAYI BAŞLAT"):
        # Örnek bir liste (Kendi listenizi genişletebilirsiniz)
        liste = ["THYAO", "ASELS", "AKBNK", "BIMAS", "ALARK", "EREGL"]
        bulunanlar = []
        with st.spinner("Taranıyor..."):
            for h in liste:
                try:
                    df = yf.download(f"{h}.IS", period="20d", progress=False)
                    if not df.empty:
                        rsi = ta.momentum.rsi(df['Close'].squeeze(), window=14).iloc[-1]
                        if rsi < 35: bulunanlar.append(h)
                except: continue
        st.write("Sinyal Üretenler:", bulunanlar if bulunanlar else "Bulunamadı.")

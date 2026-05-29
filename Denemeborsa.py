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

# Arayüz Ayarları
st.set_page_config(page_title="Mobil Borsa", layout="centered")

# Veritabanı
conn = sqlite3.connect("takip_listesi.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS watchlist (hisse_kodu TEXT PRIMARY KEY, maliyet REAL, adet INTEGER)")
conn.commit()

# --- 1. Portföy Sekmesi ---
def portfoy_sekmesi():
    st.subheader("💼 Portföy & Stop")
    kod = st.text_input("Hisse Kodu (Örn: THYAO)").upper().strip()
    mal = st.number_input("Maliyet", value=0.0)
    adet = st.number_input("Adet", value=0, step=1)
    
    if st.button("Kaydet / Güncelle"):
        if kod:
            cursor.execute("INSERT OR REPLACE INTO watchlist VALUES (?, ?, ?)", (kod, mal, adet))
            conn.commit()
            st.rerun()
            
    st.divider()
    for row in cursor.execute("SELECT * FROM watchlist").fetchall():
        c1, c2 = st.columns([3, 1])
        c1.write(f"**{row[0]}** | M: {row[1]} | A: {row[2]}")
        if c2.button("Sil", key=f"del_{row[0]}"):
            cursor.execute("DELETE FROM watchlist WHERE hisse_kodu = ?", (row[0],))
            conn.commit()
            st.rerun()

# --- 2. Analiz Sekmesi ---
def analiz_sekmesi():
    st.subheader("🔍 Hisse Analiz")
    kod = st.text_input("Hisse Kodu Giriniz:").upper().strip()
    if kod:
        df = yf.download(f"{kod}.IS", period="60d", progress=False)
        if not df.empty:
            son_fiyat = df['Close'].iloc[-1].item()
            st.metric(label=f"{kod} Fiyat", value=f"{son_fiyat:.2f} TL")
            if st.button("Listeye Ekle"):
                cursor.execute("INSERT OR REPLACE INTO watchlist VALUES (?, ?, ?)", (kod, 0.0, 0))
                conn.commit()
                st.success("Listeye eklendi!")

# --- 3. Radar Sekmesi ---
def radar_sekmesi():
    st.subheader("🔍 Mega Radar")
    if st.button("Taramayı Başlat"):
        liste = ["THYAO", "ASELS", "AKBNK", "BIMAS"]
        for h in liste:
            try:
                df = yf.download(f"{h}.IS", period="30d", progress=False)
                rsi = ta.momentum.rsi(df['Close'].squeeze(), window=14).iloc[-1]
                if rsi < 35: st.write(f"✅ {h} (RSI: {rsi:.1f})")
            except: continue

# --- Ana Gövde ---
st.title("📱 Mobil Borsa Paneli")
sekme1, sekme2, sekme3 = st.tabs(["PORTFÖY & STOP", "HİSSE ANALİZ", "MEGA RADAR"])

with sekme1: portfoy_sekmesi()
with sekme2: analiz_sekmesi()
with sekme3: radar_sekmesi()

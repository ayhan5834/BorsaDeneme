# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:42:45 2026

@author: EmirAysu
"""

import os
import sys
import logging
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    div[data-testid="stMetricWidget"] { background-color: #1E1E1E; border: 1px solid #2D2D2D; padding: 10px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. VERİTABANI SINIFI
# ==============================================================================
class Veritabani:
    def __init__(self):
        self.baglanti = sqlite3.connect("takip_listesi.db", check_same_thread=False)
        self.cursor = self.baglanti.cursor()
        self.tablo_olustur()

    def tablo_olustur(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hisse_kodu TEXT UNIQUE,
                maliyet REAL DEFAULT 0,
                adet INTEGER DEFAULT 0
            )
        """)
        self.baglanti.commit()

    def hisse_ekle(self, kod, maliyet=0.0, adet=0):
        try:
            self.cursor.execute("INSERT INTO watchlist (hisse_kodu, maliyet, adet) VALUES (?, ?, ?)", (kod, maliyet, adet))
        except sqlite3.IntegrityError:
            self.cursor.execute("UPDATE watchlist SET maliyet = ?, adet = ? WHERE hisse_kodu = ?", (maliyet, adet, kod))
        self.baglanti.commit()

    def hisse_sil(self, kod):
        self.cursor.execute("DELETE FROM watchlist WHERE hisse_kodu = ?", (kod,))
        self.baglanti.commit()

    def listeyi_getir(self):
        self.cursor.execute("SELECT hisse_kodu, maliyet, adet FROM watchlist")
        return self.cursor.fetchall()

# ==============================================================================
# 2. DİNAMİK BIST LİSTESİ MOTORU
# ==============================================================================
@st.cache_data(ttl=3600)
def dinamik_bist_listesi_yukle():
    csv_yolu = "bist_hisseler.csv"
    if os.path.exists(csv_yolu):
        df = pd.read_csv(csv_yolu)
        return df["kod"].tolist()
    return ["A1CAP", "ADEL", "AGROT", "AKBNK", "ALARK", "ASELS", "BIMAS", "THYAO"]

# ==============================================================================
# 3. YAPAY ZEKA TAHMİN MOTORU
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
# 4. STREAMLIT ARAYÜZÜ
# ==============================================================================
st.title("📱 Mobil Borsa")
db = Veritabani()
sekme1, sekme2, sekme3 = st.tabs(["PORTFÖY & STOP", "HİSSE ANALİZ", "MEGA RADAR"])

with sekme1:
    st.subheader("💼 Portföy & Durum")
    with st.form("ekle_form"):
        yeni_hisse = st.text_input("Hisse Kodu (örn: ASELS)").upper().strip()
        maliyet = st.number_input("Maliyet", value=0.0, step=0.1)
        adet = st.number_input("Adet", value=0, step=1)
        if st.form_submit_button("Kaydet / Güncelle"):
            if yeni_hisse:
                db.hisse_ekle(yeni_hisse, maliyet, adet)
                st.success("İşlem başarılı!")

    hisseler = db.listeyi_getir()
    for h, maliyet, adet in hisseler:
        with st.container(border=True):
            st.write(f"**{h}** | Maliyet: {maliyet} | Adet: {adet}")
            if st.button("Sil", key=f"del_{h}"):
                db.hisse_sil(h)
                st.rerun()

with sekme2:
    st.subheader("🔍 Detaylı Hisse Analizi")
    hisse_kodu = st.text_input("Hisse Kodu:", key="analiz_input").upper().strip()
    if hisse_kodu:
        df = yf.download(f"{hisse_kodu}.IS", period="60d", progress=False)
        if not df.empty:
            son_fiyat = df['Close'].iloc[-1].item()
            hedef, _ = mobil_tahmin_motoru(df)
            st.metric(label=f"{hisse_kodu} Fiyat", value=f"{son_fiyat:.2f} TL")
            st.write(f"YZ 5 Günlük Tahmin: {hedef:.2f} TL")
            if st.button("Listeye Ekle"):
                db.hisse_ekle(hisse_kodu, 0, 0)
                st.success("Eklendi!")

with sekme3:
    st.subheader("🔍 Mega Radar")
    if st.button("🚀 Taramayı Başlat"):
        liste = dinamik_bist_listesi_yukle()
        bulunanlar = []
        for h in liste:
            try:
                df = yf.download(f"{h}.IS", period="20d", progress=False)
                if not df.empty:
                    rsi = ta.momentum.rsi(df['Close'].squeeze(), window=14).iloc[-1]
                    if rsi < 30: bulunanlar.append(h)
            except: continue
        st.write("Sinyal Üretenler:", bulunanlar)

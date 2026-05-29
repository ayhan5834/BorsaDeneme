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
import matplotlib  # ÖNCE BU
matplotlib.use('Agg') # SONRA BU
import matplotlib.pyplot as plt
import yfinance as yf
import ta
import streamlit as st
from sklearn.linear_model import HuberRegressor

# Artık alt satırlarda kodunuzu tanımlayabilirsiniz...

# Matplotlib ayarı
matplotlib.use('Agg')

# ==============================================================================
# VERİTABANI YÖNETİMİ
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

    def hisse_ekle(self, kod, maliyet, adet):
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

@st.cache_resource
def get_db():
    return Veritabani()

db = get_db()

# ==============================================================================
# YARDIMCI FONKSİYONLAR
# ==============================================================================
@st.cache_data(ttl=3600)
def dinamik_bist_listesi_yukle():
    if os.path.exists("bist_hisseler.csv"):
        return pd.read_csv("bist_hisseler.csv")["kod"].tolist()
    return ["THYAO", "ASELS", "KRDMD", "TUPRS", "EREGL", "AKBNK", "SISE", "BIMAS", "SASA", "HEKTS"]

def mobil_tahmin_motoru(df):
    if df is None or df.empty or len(df) < 5:
        return 0.0, np.zeros(5)
    try:
        data = df.tail(60).copy()
        data['gun'] = range(len(data))
        model = HuberRegressor().fit(data[['gun']], data['Close'])
        son_gun = data['gun'].iloc[-1]
        tahminler = model.predict(np.arange(son_gun + 1, son_gun + 6).reshape(-1, 1))
        return tahminler[-1], tahminler
    except:
        return df['Close'].iloc[-1], np.full(5, df['Close'].iloc[-1])

# ==============================================================================
# STREAMLIT ARAYÜZÜ
# ==============================================================================
st.set_page_config(page_title="Mobil Borsa", layout="centered")
st.markdown("""<style>.stApp {background-color: #121212; color: white;}</style>""", unsafe_allow_html=True)

st.title("📱 Mobil Borsa")
sekme1, sekme2, sekme3 = st.tabs(["PORTFÖY", "ANALİZ", "RADAR"])

with sekme1:
    st.subheader("💼 Portföy")
    with st.expander("➕ Hisse Ekle / Güncelle"):
        yeni_hisse = st.text_input("Hisse Kodu").upper().strip()
        maliyet = st.number_input("Maliyet", value=0.0)
        adet = st.number_input("Adet", value=0)
        if st.button("Kaydet"):
            if yeni_hisse:
                db.hisse_ekle(yeni_hisse, maliyet, adet)
                st.rerun()

    hisseler = db.listeyi_getir()
    for h, m, a in hisseler:
        c1, c2 = st.columns([3, 1])
        c1.write(f"**{h}** - Adet: {a} | Maliyet: {m}")
        if c2.button("Sil", key=f"del_{h}"):
            db.hisse_sil(h)
            st.rerun()

with sekme2:
    hisse_kodu = st.text_input("Analiz için kod girin").upper().strip()
    if hisse_kodu:
        df = yf.download(f"{hisse_kodu}.IS", period="60d", progress=False)
        if not df.empty:
            son_fiyat = df['Close'].iloc[-1]
            hedef, _ = mobil_tahmin_motoru(df)
            st.metric(hisse_kodu, f"{son_fiyat:.2f} TL", f"Tahmin: {hedef:.2f} TL")
            
with sekme3:
    if st.button("Taramayı Başlat"):
        for h in dinamik_bist_listesi_yukle():
            st.write(f"Taranıyor: {h}")
            # Tarama mantığınız buraya...

# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:42:45 2026

@author: EmirAysu
"""

import os
import sys
import logging

# PyInstaller çevre değişkeni ayarı (Qt çakışmalarını önler)
if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
    qt_plugin_path = os.path.join(base_dir, "PyQt5", "Qt5", "plugins")
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_plugin_path

# Matplotlib arkada harici pencere açmasını engeller ve logları kapatır
import matplotlib
matplotlib.use('Agg') 
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# STANDART KÜTÜPHANELER
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import yfinance as yf
import ta
from sklearn.linear_model import HuberRegressor

IS_STREAMLIT = "streamlit" in sys.modules

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
            self.baglanti.commit()
            return True
        except sqlite3.IntegrityError:
            self.cursor.execute("UPDATE watchlist SET maliyet = ?, adet = ? WHERE hisse_kodu = ?", (maliyet, adet, kod))
            self.baglanti.commit()
            return True

    def hisse_sil(self, kod):
        self.cursor.execute("DELETE FROM watchlist WHERE hisse_kodu = ?", (kod,))
        self.baglanti.commit()

    def listeyi_getir(self):
        self.cursor.execute("SELECT hisse_kodu, maliyet, adet FROM watchlist")
        return self.cursor.fetchall()

# ==============================================================================
# 2. YAPAY ZEKA TAHMİN MOTORU
# ==============================================================================
def mobil_tahmin_motoru(df):
    if df is None or df.empty or len(df) < 5:
        return 0.0, np.zeros(5)
    try:
        data = df.tail(60).copy()
        data['gun'] = range(len(data))
        model = HuberRegressor(max_iter=1000)
        model.fit(data[['gun']], data['Close'].squeeze())
        son_gun_index = data['gun'].iloc[-1]
        tahmin_serisi = model.predict(np.arange(son_gun_index + 1, son_gun_index + 6).reshape(-1, 1))
        return tahmin_serisi[-1], tahmin_serisi
    except:
        return df['Close'].squeeze().iloc[-1], np.full(5, df['Close'].squeeze().iloc[-1])

# ==============================================================================
# 3. STREAMLIT PANELİ
# ==============================================================================
if IS_STREAMLIT:
    st.set_page_config(page_title="Mobil Borsa", layout="centered")
    db = Veritabani()
    sekme1, sekme2, sekme3 = st.tabs(["PORTFÖY", "ANALİZ", "RADAR"])

    with sekme2:
        st.subheader("🔍 Detaylı Hisse Analizi")
        hisse_kodu = st.text_input("Hisse Kodu Giriniz", key="a_kod").upper().strip()
        
        if st.button("Analiz Et", key="btn_analiz") and hisse_kodu:
            sorgu_kodu = f"{hisse_kodu}.IS"
            df = yf.download(sorgu_kodu, period="60d", interval="1d", progress=False)
            if not df.empty:
                kapanis = df['Close'].squeeze()
                hacim = df['Volume'].squeeze()
                
                # İndikatörler
                son_rsi = ta.momentum.rsi(kapanis, window=14).iloc[-1]
                macd_obj = ta.trend.MACD(kapanis)
                hacim_ort = hacim.rolling(10).mean().iloc[-1]
                hacim_onay = hacim.iloc[-1] > (hacim_ort * 0.8)
                
                durum = "GÜÇLÜ AL" if (son_rsi < 42 and hacim_onay) else ("DİKKAT (Hacimsiz)" if son_rsi < 42 else "TUT/SAT")
                st.write(f"### Durum: {durum} | Hacim Onay: {'✅' if hacim_onay else '❌'}")
                
                # --- GRAFİK ÇİZİMİ ---
                fig, ax = plt.subplots(figsize=(6, 3), facecolor='#121212')
                ax.set_facecolor('#1E1E1E')
                ax.plot(kapanis.tail(30).values, color='#00F0FF', label="Fiyat")
                ax.tick_params(colors='white')
                st.pyplot(fig)
                
                hedef, _ = mobil_tahmin_motoru(df)
                st.write(f"YZ Tahmin: {hedef:.2f} TL")

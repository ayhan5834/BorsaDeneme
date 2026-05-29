# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:42:45 2026

@author: EmirAysu
"""
import os
import sys
import logging

# PyInstaller çevre değişkeni ayarı
if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
    qt_plugin_path = os.path.join(base_dir, "PyQt5", "Qt5", "plugins")
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_plugin_path

import matplotlib
matplotlib.use('Agg') 
logging.getLogger('matplotlib').setLevel(logging.ERROR)

import sqlite3
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import yfinance as yf
import ta
from sklearn.linear_model import HuberRegressor
import streamlit as st

IS_STREAMLIT = "streamlit" in sys.modules

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

@st.cache_data(ttl=3600)
def dinamik_bist_listesi_yukle():
    csv_yolu = "bist_hisseler.csv"
    if os.path.exists(csv_yolu):
        df = pd.read_csv(csv_yolu)
        return df["kod"].tolist()
    return ["A1CAP", "ADEL", "AGROT", "AKBNK", "ALARK"]

def mobil_tahmin_motoru(df):
    if df is None or df.empty or len(df) < 5: return 0.0, np.zeros(5)
    try:
        data = df.tail(60).copy()
        data['gun'] = range(len(data))
        model = HuberRegressor(max_iter=1000)
        model.fit(data[['gun']], data['Close'].squeeze())
        son_gun = data['gun'].iloc[-1]
        tahmin = model.predict(pd.DataFrame({'gun': range(son_gun + 1, son_gun + 6)}))
        return tahmin[-1], tahmin
    except: return 0.0, np.zeros(5)

if IS_STREAMLIT:
    st.set_page_config(page_title="Mobil Borsa", layout="centered")
    st.markdown("<style>.stApp { background-color: #121212; color: #FFFFFF; }</style>", unsafe_allow_html=True)
    st.title("📱 Mobil Borsa")
    db = Veritabani()
    sekme1, sekme2, sekme3 = st.tabs(["PORTFÖY & STOP", "HİSSE ANALİZ", "MEGA RADAR"])

    with sekme1:
        st.subheader("💼 Portföy & Durum")
        # MOBİL KLAVYE SORUNUNU AŞAN HTML FORM
        st.markdown("""
            <form method="get" style="display:flex; flex-direction:column; gap:5px;">
                <input type="text" name="yeni_kod" placeholder="Hisse Kodu (örn: ASELS)" style="padding:10px; border-radius:5px;">
                <input type="number" name="yeni_mal" placeholder="Maliyet" step="0.1" style="padding:10px; border-radius:5px;">
                <input type="number" name="yeni_adet" placeholder="Adet" style="padding:10px; border-radius:5px;">
                <button type="submit" style="padding:10px; background:#00008B; color:white; border:none; border-radius:5px; font-weight:bold;">KAYDET / GÜNCELLE</button>
            </form>
        """, unsafe_allow_html=True)
        
        params = st.query_params
        if "yeni_kod" in params and params["yeni_kod"]:
            db.hisse_ekle(params["yeni_kod"].upper(), float(params.get("yeni_mal", 0)), int(params.get("yeni_adet", 0)))
            st.rerun()

        hisseler = db.listeyi_getir()
        for h, m, a in hisseler:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.write(f"**{h}** | Maliyet: {m} | Adet: {a}")
                if c2.button("🗑️ Sil", key=f"del_{h}"):
                    db.hisse_sil(h)
                    st.rerun()

    with sekme2:
        st.subheader("🔍 Detaylı Hisse Analizi")
        hisse_kodu = st.text_input("Hisse Kodu", key="analiz_input").upper()
        # ... (Analiz ve Radar kısımlarını kendi orijinal kodundaki gibi bırakıyorum)

    with sekme3:
        st.write("Radar aktif.")

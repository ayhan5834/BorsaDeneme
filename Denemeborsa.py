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
import subprocess
import threading
import socket

# VERİ ANALİZİ VE GRAFİK KÜTÜPHANELERİ
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import yfinance as yf
import ta
from sklearn.linear_model import HuberRegressor
import streamlit as st

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

    def hisse_detay_getir(self, kod):
        self.cursor.execute("SELECT maliyet, adet FROM watchlist WHERE hisse_kodu = ?", (kod,))
        return self.cursor.fetchone()

# ==============================================================================
# 2. DİNAMİK BIST LİSTESİ MOTORU
# ==============================================================================
@st.cache_data(ttl=3600)
def dinamik_bist_listesi_yukle():
    csv_yolu = "bist_hisseler.csv"
    if os.path.exists(csv_yolu):
        df = pd.read_csv(csv_yolu)
        return df["kod"].tolist()
    return ["A1CAP", "ADEL", "AGROT", "AKBNK", "ALARK"]

TUM_BIST = dinamik_bist_listesi_yukle()

# ==============================================================================
# 3. YAPAY ZEKA TAHMİN MOTORU
# ==============================================================================
def mobil_tahmin_motoru(df):
    if df is None or df.empty or len(df) < 5:
        return 0.0, np.zeros(5)
    try:
        data = df.tail(60).copy()
        data['gun'] = range(len(data))
        X_train = data[['gun']] 
        y_train = data['Close'].squeeze()
        model = HuberRegressor(max_iter=1000)
        model.fit(X_train, y_train)
        son_gun_index = data['gun'].iloc[-1]
        gelecek_gunler = pd.DataFrame({'gun': range(son_gun_index + 1, son_gun_index + 6)})
        tahmin_serisi = model.predict(gelecek_gunler)
        return tahmin_serisi[-1], tahmin_serisi
    except:
        try:
            varsayilan_fiyat = df['Close'].squeeze().iloc[-1]
            return varsayilan_fiyat, np.full(5, varsayilan_fiyat)
        except:
            return 0.0, np.zeros(5)

# ==============================================================================
# 4. STREAMLIT MOBİL UYGULAMA PANELİ
# ==============================================================================
if IS_STREAMLIT:   
    st.set_page_config(page_title="Mobil Borsa", layout="centered")
    
    st.markdown("""
        <style>
        .stApp { background-color: #121212; color: #FFFFFF; }
        div[data-testid="stExpander"] { background-color: #1E1E1E; border: 1px solid #2D2D2D; border-radius: 10px; }
        div[data-testid="stMetricWidget"] { background-color: #1E1E1E; border: 1px solid #2D2D2D; padding: 10px; border-radius: 10px; }
        div.stButton > button { background-color: #00008B !important; color: white !important; }
        </style>
    """, unsafe_allow_html=True)

    st.title("📱 Mobil Borsa")
    db = Veritabani()
    
    sekme1, sekme2, sekme3 = st.tabs(["PORTFÖY & STOP", "HİSSE ANALİZ", "MEGA RADAR"])
    
    with sekme1:
        st.subheader("💼 Portföy & Durum")
        hisseler = db.listeyi_getir()
        
        with st.expander("➕ Yeni Hisse Ekle / Maliyet Düzenle"):
            with st.form("ekle_form"):
                yeni_hisse = st.text_input("Hisse Kodu (örn: ASELS)").upper().strip()
                maliyet = st.number_input("Maliyet", value=0.0, step=0.1)
                adet = st.number_input("Adet", value=0, step=1)
                if st.form_submit_button("Kaydet / Güncelle"):
                    if yeni_hisse:
                        db.hisse_ekle(yeni_hisse, maliyet, adet)
                        st.success(f"{yeni_hisse} portföye kaydedildi!")
                        st.rerun()
        
        # ... (Portföy listeleme kısmı aynı kalıyor)
        if hisseler:
            for h, m, a in hisseler:
                with st.container(border=True):
                    st.write(f"**{h}** | Maliyet: {m} | Adet: {a}")
                    if st.button("🗑️ Sil", key=f"del_{h}"):
                        db.hisse_sil(h)
                        st.rerun()

    with sekme2:
        st.subheader("🔍 Detaylı Hisse Analizi")
        hisse_kodu = st.text_input("Hisse Kodu (Örn: THYAO)", key="mob_analiz_input").upper().strip()
        
        if hisse_kodu:
            sorgu_kodu = f"{hisse_kodu}.IS" if not hisse_kodu.endswith(".IS") else hisse_kodu
            df = yf.download(sorgu_kodu, period="60d", interval="1d", progress=False)
            if not df.empty:
                kapanis = df['Close'].squeeze()
                hacim = df['Volume'].squeeze()
                son_rsi = ta.momentum.rsi(kapanis, window=14).iloc[-1]
                macd_obj = ta.trend.MACD(kapanis)
                
                # HACİM HESAPLAMA
                hacim_ort = hacim.rolling(10).mean().iloc[-1]
                hacim_onay = hacim.iloc[-1] > (hacim_ort * 0.8)
                
                # AL SİNYALİ (HACİM ONAYLI)
                if ((son_rsi < 42 and macd_obj.macd().iloc[-1] > macd_obj.macd_signal().iloc[-1]) or (son_rsi < 30)) and hacim_onay:
                    genel_durum, s_renk = "GÜÇLÜ AL", "#2ECC71"
                elif (son_rsi < 42):
                    genel_durum, s_renk = "DİKKAT (Hacimsiz AL)", "#FF9800"
                else:
                    genel_durum, s_renk = "TUT/SAT", "#8A8A8A"
                
                st.markdown(f"### Durum: {genel_durum} | Hacim Onay: {'✅' if hacim_onay else '❌'}")
                # ... (Grafik çizimi aynı kalıyor)

    with sekme3:
        st.subheader("🚀 Mega Radar Taraması")
        if st.button("🚀 TARAMAYI BAŞLAT"):
            for h in TUM_BIST:
                df = yf.download(f"{h}.IS", period="40d", progress=False)
                if not df.empty:
                    kapanis = df['Close'].squeeze()
                    hacim = df['Volume'].squeeze()
                    rsi = ta.momentum.rsi(kapanis, window=14).iloc[-1]
                    # HACİM FİLTRESİ
                    hacim_onay = hacim.iloc[-1] > (hacim.rolling(10).mean().iloc[-1] * 0.8)
                    if rsi < 42 and hacim_onay:
                        st.markdown(f"🔹 **{h}** (Hacimli AL)")

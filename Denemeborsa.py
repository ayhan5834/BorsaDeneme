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
import matplotlib
import matplotlib.pyplot as plt
import yfinance as yf
import ta
import streamlit as st
from sklearn.linear_model import HuberRegressor

# Matplotlib ayarı
matplotlib.use('Agg')
logging.getLogger('matplotlib').setLevel(logging.ERROR)

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

# Veritabanını sabitleyen yapı (Kilitlenmeyi önler)
@st.cache_resource
def get_db():
    return Veritabani()

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
        son_gun_index = data['gun'].iloc[-1]
        gelecek_gunler = pd.DataFrame({'gun': range(son_gun_index + 1, son_gun_index + 6)})
        tahmin_serisi = model.predict(gelecek_gunler)
        return tahmin_serisi[-1], tahmin_serisi
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
        </style>
    """, unsafe_allow_html=True)

    st.title("📱 Mobil Borsa")
    db = get_db() # Sabitlenmiş bağlantıyı kullan
    
    sekme1, sekme2, sekme3 = st.tabs(["PORTFÖY & STOP", "HİSSE ANALİZ", "MEGA RADAR"])
    
    with sekme1:
        st.subheader("💼 Portföy & Durum")
        
        with st.expander("➕ Yeni Hisse Ekle / Maliyet Düzenle"):
            yeni_hisse = st.text_input("Hisse Kodu", key="mob_ekle_kod").upper().strip()
            maliyet = st.number_input("Maliyet", value=0.0, step=0.1, key="mob_ekle_mal")
            adet = st.number_input("Adet", value=0, step=1, key="mob_ekle_adet")
            
            if st.button("Kaydet / Güncelle", key="mob_kaydet_btn"):
                if yeni_hisse:
                    db.hisse_ekle(yeni_hisse, maliyet, adet)
                    st.success(f"{yeni_hisse} portföye kaydedildi!")
                    st.rerun() # Arayüzü hemen güncelle
        
        hisseler = db.listeyi_getir()
        if not hisseler:
            st.warning("Henüz takip listesinde hisse yok.")
        else:
            for h, maliyet, adet in hisseler:
                col1, col2 = st.columns([3, 1])
                col1.write(f"**{h}** | Maliyet: {maliyet} | Adet: {adet}")
                if col2.button("🗑️ Sil", key=f"del_{h}"):
                    db.hisse_sil(h)
                    st.rerun()

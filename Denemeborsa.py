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

# Ayarlar
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
def get_db(): return Veritabani()

# ==============================================================================
# 2. MOBİL UYUMLU VERİ GİRİŞ PANELİ (HTML - KOYU MAVİ BUTON)
# ==============================================================================
def mobil_veri_giris_formu(db):
    st.markdown("""
        <form method="get" style="background:#1E1E1E; padding:15px; border-radius:10px; border:1px solid #2D2D2D;">
            <input type="text" name="kod" placeholder="Hisse Kodu (Örn: THYAO)" style="width:100%; padding:10px; margin-bottom:10px; border-radius:5px; border:none; background:#2D2D2D; color:white;">
            <input type="number" name="maliyet" placeholder="Maliyet" step="0.01" style="width:100%; padding:10px; margin-bottom:10px; border-radius:5px; border:none; background:#2D2D2D; color:white;">
            <input type="number" name="adet" placeholder="Adet" style="width:100%; padding:10px; margin-bottom:10px; border-radius:5px; border:none; background:#2D2D2D; color:white;">
            <button type="submit" style="width:100%; padding:10px; background:#00008B; color:white; border:none; border-radius:5px; font-weight:bold;">KAYDET</button>
        </form>
    """, unsafe_allow_html=True)
    
    params = st.query_params
    if "kod" in params and params["kod"]:
        kod = params["kod"].upper().strip()
        maliyet = float(params["maliyet"]) if params["maliyet"] else 0.0
        adet = int(params["adet"]) if params["adet"] else 0
        db.hisse_ekle(kod, maliyet, adet)
        st.success(f"{kod} başarıyla kaydedildi!")
        st.rerun()

# ==============================================================================
# 3. ANA UYGULAMA
# ==============================================================================
if IS_STREAMLIT:
    st.set_page_config(page_title="Mobil Borsa", layout="centered")
    db = get_db()
    st.title("📱 Mobil Borsa")
    
    sekme1, sekme2, sekme3 = st.tabs(["PORTFÖY", "ANALİZ", "RADAR"])
    
    with sekme1:
        st.subheader("💼 Portföy Yönetimi")
        mobil_veri_giris_formu(db)
        
        hisseler = db.listeyi_getir()
        for h, m, a in hisseler:
            c1, c2 = st.columns([3, 1])
            c1.write(f"**{h}** | Maliyet: {m} | Adet: {a}")
            # Sil butonu Streamlit'in kendi butonu olduğu için buna renk müdahalesi 
            # sadece tema ayarlarından olur, kod ile burayı değiştirmedim.
            if c2.button("Sil", key=f"del_{h}"):
                db.hisse_sil(h)
                st.rerun()

    with sekme2:
        st.write("Analiz sekmesi aktif.")
    
    with sekme3:
        st.write("Radar sekmesi aktif.")

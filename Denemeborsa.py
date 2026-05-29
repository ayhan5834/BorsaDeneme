# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:42:45 2026

@author: EmirAysu
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yfinance as yf
import ta
import streamlit as st
from sklearn.linear_model import HuberRegressor

# Temel ayarlar
matplotlib.use('Agg')
st.set_page_config(page_title="Mobil Borsa", layout="centered")

# CSS: Mobil cihazlarda input kutularının tıklanabilirliğini artırır
st.markdown("""
    <style>
    div[data-testid="stTextInput"] input {
        font-size: 16px !important; /* Mobil için en iyi okuma boyutu */
        padding: 10px !important;
    }
    .stApp { background-color: #121212; color: #FFFFFF; }
    </style>
""", unsafe_allow_html=True)

# Veritabanı Yönetimi
class Veritabani:
    def __init__(self):
        self.baglanti = sqlite3.connect("takip_listesi.db", check_same_thread=False)
        self.cursor = self.baglanti.cursor()
        self.cursor.execute("CREATE TABLE IF NOT EXISTS watchlist (id INTEGER PRIMARY KEY, hisse_kodu TEXT UNIQUE, maliyet REAL, adet INTEGER)")
        self.baglanti.commit()

    def hisse_ekle(self, kod, maliyet, adet):
        try:
            self.cursor.execute("INSERT OR REPLACE INTO watchlist (hisse_kodu, maliyet, adet) VALUES (?, ?, ?)", (kod, maliyet, adet))
            self.baglanti.commit()
        except: pass

    def listeyi_getir(self):
        return self.cursor.execute("SELECT hisse_kodu, maliyet, adet FROM watchlist").fetchall()

@st.cache_resource
def get_db(): return Veritabani()

db = get_db()

# Arayüz
st.title("📱 Mobil Borsa")
yeni_hisse = st.text_input("Hisse Kodu (Örn: ASELS)", key="kod")
maliyet = st.number_input("Maliyet", value=0.0, key="mal")
adet = st.number_input("Adet", value=0, key="adet")

if st.button("Kaydet"):
    if yeni_hisse:
        db.hisse_ekle(yeni_hisse.upper(), maliyet, adet)
        st.success("Kaydedildi!")
        st.rerun()

st.write("---")
for h, m, a in db.listeyi_getir():
    st.write(f"**{h}** - Adet: {a} | Maliyet: {m}")

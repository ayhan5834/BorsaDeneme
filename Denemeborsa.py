# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:42:45 2026

@author: EmirAysu
"""

import streamlit as st
import sqlite3
import os

# 1. Veritabanı Yapılandırması
class Veritabani:
    def __init__(self):
        self.baglanti = sqlite3.connect("takip_listesi.db", check_same_thread=False)
        self.cursor = self.baglanti.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                hisse_kodu TEXT PRIMARY KEY,
                maliyet REAL,
                adet INTEGER
            )
        """)
        self.baglanti.commit()

    def hisse_ekle(self, kod, maliyet, adet):
        self.cursor.execute("INSERT OR REPLACE INTO watchlist VALUES (?, ?, ?)", (kod, maliyet, adet))
        self.baglanti.commit()

    def listeyi_getir(self):
        return self.cursor.execute("SELECT * FROM watchlist").fetchall()

# Veritabanını sabitle
@st.cache_resource
def get_db():
    return Veritabani()

db = get_db()

# 2. Arayüz (Sadece Veri Girişi)
st.title("Veri Giriş Paneli")

with st.form("veri_giris_formu", clear_on_submit=True):
    kod = st.text_input("Hisse Kodu").upper()
    maliyet = st.number_input("Maliyet", step=0.01)
    adet = st.number_input("Adet", step=1)
    
    submit = st.form_submit_button("Kaydet")
    if submit and kod:
        db.hisse_ekle(kod, maliyet, adet)
        st.success(f"{kod} kaydedildi!")

# 3. Liste Gösterimi
st.subheader("Kayıtlı Hisseler")
for item in db.listeyi_getir():
    st.write(f"Hisse: {item[0]} | Maliyet: {item[1]} | Adet: {item[2]}")

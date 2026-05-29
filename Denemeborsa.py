# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:42:45 2026

@author: EmirAysu
"""

import streamlit as st
import sqlite3

# Veritabanı bağlantısı
def get_db():
    conn = sqlite3.connect("takip_listesi.db", check_same_thread=False)
    return conn

conn = get_db()
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS watchlist (hisse TEXT PRIMARY KEY)")
conn.commit()

st.title("Mobil Veri Girişi")

# Yazılabilir olan HTML formumuzu ekleyelim
html_form = """
<form method="get" style="display:flex; flex-direction:column; gap:10px;">
  <input type="text" name="hisse_adi" placeholder="Hisse kodunu yazın..." 
         style="width: 100%; height: 50px; font-size: 20px; padding: 10px; border-radius: 8px; border: 1px solid #ccc;">
  <button type="submit" style="height: 50px; font-size: 18px; border-radius: 8px;">Kaydet</button>
</form>
"""
st.components.v1.html(html_form, height=150)

# Gönderilen veriyi veritabanına ekle
query_params = st.query_params
if "hisse_adi" in query_params:
    hisse = query_params["hisse_adi"].upper()
    try:
        cursor.execute("INSERT INTO watchlist (hisse) VALUES (?)", (hisse,))
        conn.commit()
        st.success(f"{hisse} başarıyla veritabanına kaydedildi!")
    except:
        st.warning(f"{hisse} zaten kayıtlı.")

# Listeleme
st.subheader("Kayıtlı Hisseler:")
cursor.execute("SELECT * FROM watchlist")
hisseler = cursor.fetchall()
for h in hisseler:
    st.write(f"✅ {h[0]}")

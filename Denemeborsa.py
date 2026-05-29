# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:42:45 2026

@author: EmirAysu
"""

import streamlit as st

st.title("Mobil Klavye Testi")

# Veritabanı yok, sadece kutular var
# Mobil klavyenin kutuyu engellememesi için özel stil
st.markdown("""
    <style>
    input {
        font-size: 16px !important;
        padding: 15px !important;
    }
    </style>
""", unsafe_allow_html=True)

kod = st.text_input("Buraya bir şeyler yazmayı dene")
maliyet = st.number_input("Rakam girmeyi dene", value=0.0)

if kod:
    st.write(f"Şu an şunları yazdın: {kod}")

# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:42:45 2026

@author: EmirAysu
"""

import streamlit as st

st.title("Test Paneli")

# text_input yerine text_area deniyoruz
yazi = st.text_area("Buraya yazmayı dene (text_area)", height=100)

if st.button("Kontrol Et"):
    st.write("Yazdığın metin: ", yazi)

# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:42:45 2026

@author: EmirAysu
"""

import streamlit as st

st.title("Alternatif Giriş Paneli")

# Saf HTML formu - Streamlit'in kendi text_input'unu kullanmıyoruz
# Bu yöntem mobil tarayıcıların "native" klavye tetikleyicisini kullanır
html_form = """
<form action="" method="get">
  <input type="text" name="hisse" placeholder="Hisse kodunu buraya yazın..." 
         style="width: 100%; height: 50px; font-size: 20px; padding: 10px; border: 2px solid #ccc; border-radius: 8px;">
  <input type="submit" value="Gönder" style="margin-top: 10px; width: 100%; height: 50px; font-size: 18px;">
</form>
"""

st.components.v1.html(html_form, height=150)

# Gönderilen veriyi yakala
query_params = st.query_params
if "hisse" in query_params:
    st.write(f"Yakalanan veri: {query_params['hisse']}")

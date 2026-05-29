# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:42:45 2026

@author: EmirAysu
"""


import streamlit as st
import pandas as pd

# 1. Sayfa Ayarlarını Geniş Mod Yapıyoruz
st.set_page_config(
    page_title="Masaüstü Görünümlü Takip Sistemi", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# 2. Mobil Tarayıcıya Masaüstü Genişliğini Dayatan CSS
st.markdown(
    """
    <style>
        /* Ekran ne kadar küçük olursa olsun sayfayı 1250 piksele sabitler */
        .main .block-container {
            min-width: 1250px !important;
            max-width: 1250px !important;
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            margin: 0 auto !important;
        }
        
        /* Mobilde sütunların alt alta kaymasını kesin olarak engeller */
        [data-testid="column"] {
            width: calc(100% / var(--array-length, 1) - 1rem) !important;
            flex: 1 1 calc(100% / var(--array-length, 1) - 1rem) !important;
            min-width: 100px !important;
        }
        
        /* Görsel estetik: PyQt QSS havası katmak için buton ve kutu tasarımları */
        .stButton>button {
            border-radius: 6px;
            height: 40px;
            width: 100%;
        }
        div[data-testid="stForm"] {
            border-radius: 8px;
            border: 1px solid #ddd;
            padding: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 3. Örnek Veri Yönetimi (Session State)
if "takip_listesi" not in st.session_state:
    st.session_state.takip_listesi = [
        {"Kod": "THYAO", "Adı": "Türk Hava Yolları", "Adet": 100, "Fiyat": 310.50},
        {"Kod": "EREGL", "Adı": "Ereğli Demir Çelik", "Adet": 250, "Fiyat": 52.20},
        {"Kod": "ASELS", "Adı": "Aselsan", "Adet": 150, "Fiyat": 64.10}
    ]

# --- ARAYÜZ BAŞLIYOR (Masaüstü Düzeni) ---
st.title("📊 Takip Listesi ve Stok Yönetim Paneli")
st.write("Bilgisayar görünümüdür. Mobilde parmaklarınızla yakınlaştırarak (zoom) kullanabilirsiniz.")

# Yan yana 2 ana sütun oluşturuyoruz (Masaüstü tasarımı)
sol_kolon, sag_kolon = st.columns([1, 2], gap="large")

# --- SOL KOLON: EKLEME / SİLME FORMU ---
with sol_kolon:
    st.subheader("🛠️ İşlemler")
    
    with st.form("ekleme_formu", clear_on_submit=True):
        st.write("**Yeni Eleman Ekle**")
        kod = st.text_input("Hisse / Stok Kodu:").upper()
        isim = st.text_input("Ürün / Şirket Adı:")
        adet = st.number_input("Adet / Miktar:", min_value=0, value=0, step=1)
        fiyat = st.number_input("Birim Fiyat:", min_value=0.0, value=0.0, step=0.1)
        
        ekle_butonu = st.form_submit_button("Listeye Ekle")
        
        if ekle_butonu and kod and isim:
            st.session_state.takip_listesi.append({
                "Kod": kod, "Adı": isim, "Adet": adet, "Fiyat": fiyat
            })
            st.success(f"{kod} başarıyla eklendi!")
            st.rerun()

    st.write("---")
    
    # Silme İşlemi
    st.write("**Eleman Sil**")
    if st.session_state.takip_listesi:
        silinecekler = [item["Kod"] for item in st.session_state.takip_listesi]
        silinecek_kod = st.selectbox("Silmek istediğiniz kodu seçin:", silinecekler)
        if st.button("Seçileni Listeden Sil", type="primary"):
            st.session_state.takip_listesi = [i for i in st.session_state.takip_listesi if i["Kod"] != silinecek_kod]
            st.toast(f"{silinecek_kod} listeden kaldırıldı.")
            st.rerun()
    else:
        st.info("Listede silinecek eleman yok.")

# --- SAĞ KOLON: LİSTELEME VE TABLO ---
with sag_kolon:
    st.subheader("📋 Güncel Takip Listesi")
    
    if st.session_state.takip_listesi:
        # Veriyi DataFrame'e çevirip basıyoruz
        df = pd.DataFrame(st.session_state.takip_listesi)
        
        # Toplam Değer Hesaplama (Adet * Fiyat)
        df["Toplam Değer"] = df["Adet"] * df["Fiyat"]
        
        # Tabloyu ekrana basıyoruz
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Alt Toplam Bilgileri (Yan yana 3 kutu)
        st.write("---")
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric("Toplam Çeşit", len(df))
        with kpi2:
            st.metric("Toplam Adet", int(df["Adet"].sum()))
        with kpi3:
            st.metric("Genel Portföy Değeri", f"{df['Toplam Değer'].sum():,.2f} TL")
            
    else:
        st.warning("Takip listeniz şu anda boş. Sol taraftan ekleme yapabilirsiniz.")

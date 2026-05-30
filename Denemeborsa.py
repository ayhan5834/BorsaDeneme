# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:42:45 2026

@author: EmirAysu
"""

import os
import logging
import sqlite3
import numpy as np  
import pandas as pd
import yfinance as yf
import ta
from sklearn.linear_model import HuberRegressor
import plotly.graph_objects as go
import streamlit as st
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import matplotlib

# Matplotlib arkada harici pencere açmasını engeller ve logları kapatır
matplotlib.use('Agg') 
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# ==============================================================================
# STREAMLIT SAYFA AYARLARI (Mobilde tam ekran deneyimi için en üste alınmalı)
# ==============================================================================
st.set_page_config(page_title="Mobil Borsa", layout="wide", initial_sidebar_state="collapsed")

# ==============================================================================
# 1. VERİTABANI SINIFI
# ==============================================================================
class Veritabani:
    def __init__(self):
        self.baglanti = sqlite3.connect("takip_listesi.db", check_same_thread=False)
        self.cursor = self.baglanti.cursor()
        self.tablo_olustur()
        if "grafik_aktif_hisse" not in st.session_state:
            st.session_state["grafik_aktif_hisse"] = None

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
    
    return ["A1CAP", "ADEL", "AGROT", "AKBNK", "ALARK", "ASELS", "THYAO"]

# --- HIZLANDIRICI ÖNBELLEK FONKSİYONLARI ---
@st.cache_data(ttl=60)  
def guncel_fiyat_indir(sorgu_kodu):
    return yf.download(sorgu_kodu, period="1d", interval="5m", progress=False)

@st.cache_data(ttl=300) 
def grafik_verisi_indir(sorgu_kodu):
    return yf.download(sorgu_kodu, period="3mo", interval="1d", progress=False)

# Canlı listeyi değişkene aktar
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
# 4. MOBİL UYGULAMA PANELİ (STREAMLIT YÜZÜ)
# ==============================================================================

# --- SMART CSS PANEL (Mobil Uyumlu) ---
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #FFFFFF; }
    div[data-testid="stExpander"] { background-color: #1E1E1E; border: 1px solid #2D2D2D; border-radius: 10px; }
    
    /* Büyük Aksiyon Butonları */
    div.stFormSubmitButton > button {
        background-color: #007BFF !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        transition: 0.3s !important;
        width: 100% !important;
    }
    
    /* Tablo ve Küçük Menü Standart Butonları */
    div.stButton > button {
        background-color: #007BFF !important;
        color: white !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 2px 10px !important;
        font-weight: bold !important;
        font-size: 13px !important;
        height: 28px !important;
        line-height: 1.2 !important;
        transition: 0.3s !important;
    }
    
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background-color: #0056b3 !important;
        color: #ffffff !important;
    }

    div[data-testid="stForm"] { background: transparent; border: none; padding: 0; }
    
    /* Giriş alanının üstündeki açıklama yazısını beyaz yapar */
    div[data-testid="stTextInput"] label, div[data-testid="stTextInput"] label p {
        color: #FFFFFF !important;
    }
    
    /* Checkbox yazılarını beyaz yapar */
    div[data-testid="stCheckbox"] label, div[data-testid="stCheckbox"] p {
        color: #FFFFFF !important;
    }
    
    /* Popover menü tasarımı */
    div[data-testid="stPopover"] button {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        color: #FFFFFF !important;
        text-align: left !important;
        padding: 10px 0px !important;
        width: 100% !important;
        border-radius: 0 !important;
        font-size: 14px !important;
        transition: none !important;
    }
    div[data-testid="stPopover"] button:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #00F0FF !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📱 Mobil Borsa")
db = Veritabani()

# Oturum Durum Yönetimleri
if "menü_aktif_hisse" not in st.session_state:
    st.session_state["menü_aktif_hisse"] = None
if "grafik_goster" not in st.session_state:
    st.session_state["grafik_goster"] = False
if "analiz_edilen_hisse" not in st.session_state:
    st.session_state["analiz_edilen_hisse"] = ""
    
def menü_tetikleyici(hisse_adi):
    if st.session_state["menü_aktif_hisse"] == hisse_adi:
        st.session_state["menü_aktif_hisse"] = None  
        st.session_state["grafik_goster"] = False
    else:
        st.session_state["menü_aktif_hisse"] = hisse_adi 
        st.session_state["grafik_goster"] = False

sekme1, sekme2, sekme3 = st.tabs(["PORTFÖY", "HİSSE ANALİZ", "RADAR"])

# --- 1. SEKME: PORTFÖY ---
with sekme1:
    hisseler = db.listeyi_getir()

    with st.expander("➕ Yeni Hisse Ekle / Düzenle"):
        with st.form(key="hisse_ekleme_formu", clear_on_submit=True):
            yeni_hisse = st.text_input("Hisse Kodu (örn: ASELS)").upper().strip()

            col_maliyet, col_adet = st.columns(2)

            maliyet = col_maliyet.number_input(
                "Maliyet",
                value=0.0,
                step=0.1
            )

            adet = col_adet.number_input(
                "Adet",
                value=0,
                step=1
            )

            kaydet_butonu = st.form_submit_button("Kaydet")

            if kaydet_butonu and yeni_hisse:
                db.hisse_ekle(yeni_hisse, maliyet, adet)
                st.rerun()

    if not hisseler:
        st.warning("Henüz takip listesinde hisse yok.")

    else:
        toplam_maliyet_hacmi = 0.0
        toplam_guncel_hacim = 0.0
        kartlar_verisi = []

        for h, maliyet, adet in hisseler:

            sorgu_kodu = h if h.endswith(".IS") else h + ".IS"

            try:
                df = guncel_fiyat_indir(sorgu_kodu)

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)

                if df is None or df.empty:
                    kartlar_verisi.append((h, 0.0, maliyet, adet, 0.0))
                    continue

                bugun_fiyat = df["Close"].squeeze().iloc[-1]

                if maliyet > 0:
                    degisim = ((bugun_fiyat - maliyet) / maliyet) * 100

                    toplam_maliyet_hacmi += maliyet * adet
                    toplam_guncel_hacim += bugun_fiyat * adet

                else:
                    dun_fiyat = (
                        df["Close"].squeeze().iloc[-2]
                        if len(df) >= 2
                        else bugun_fiyat
                    )

                    degisim = (
                        (bugun_fiyat - dun_fiyat)
                        / dun_fiyat
                    ) * 100

                kartlar_verisi.append(
                    (h, bugun_fiyat, maliyet, adet, degisim)
                )

            except:
                kartlar_verisi.append((h, 0.0, maliyet, adet, 0.0))

        if toplam_maliyet_hacmi > 0:

            toplam_kar_zarar_yuzde = (
                (toplam_guncel_hacim - toplam_maliyet_hacmi)
                / toplam_maliyet_hacmi
            ) * 100

            st.markdown(
                f"""
                <div style="
                    background-color:#1E1E1E;
                    padding:15px;
                    border-radius:10px;
                    border:1px solid #2D2D2D;
                    text-align:center;
                    margin-bottom:10px;
                ">
                    <span style="
                        color:#00F0FF;
                        font-weight:bold;
                        font-size:16px;
                    ">
                        Kasa: {toplam_maliyet_hacmi:,.2f} TL
                    </span>

                    <br>

                    <span style="
                        color:{'#2ECC71' if toplam_kar_zarar_yuzde >= 0 else '#E74C3C'};
                        font-weight:bold;
                        font-size:14px;
                    ">
                        Net Durum:
                        %{toplam_kar_zarar_yuzde:+,.2f}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Başlık satırı
        hdr_kod, hdr_fiyat, hdr_durum = st.columns([2, 2, 2])

        with hdr_kod:
            st.markdown(
                "<b style='color:#AAAAAA;'>Hisse</b>",
                unsafe_allow_html=True
            )

        with hdr_fiyat:
            st.markdown(
                "<b style='color:#AAAAAA;'>Fiyat</b>",
                unsafe_allow_html=True
            )

        with hdr_durum:
            st.markdown(
                "<b style='color:#AAAAAA;'>K/Z</b>",
                unsafe_allow_html=True
            )

        st.divider()

        for h, fiyat, m, adet, degisim in kartlar_verisi:

            fiyat_gosterim = (
                f"{fiyat:.2f}"
                if fiyat > 0
                else "--"
            )

            renk_kz = (
                "#2ECC71"
                if degisim > 0
                else "#E74C3C"
                if degisim < 0
                else "#FFFFFF"
            )

            durum_gosterim = f"%{degisim:+.2f}"

            col_kod, col_fiyat, col_durum = st.columns([2,2,2])
            with col_kod:
                c1, c2 = st.columns([4,1])

                with c1:
                    st.markdown(
                        f"<span style='font-weight:bold;color:#00F0FF'>{h}</span>",
                        unsafe_allow_html=True
                    )

                with c2:
                    with st.popover("⚙️"):

                        if st.button(
                            "Analiz/Grafik",
                            key=f"gr_{h}"
                        ):
                            st.session_state["grafik_aktif_hisse"] = (
                                None
                                if st.session_state.get("grafik_aktif_hisse") == h
                                else h
                            )
                            st.rerun()

                        if st.button(
                            "Listeden Çıkar",
                            key=f"del_{h}"
                        ):
                            db.hisse_sil(h)
                            st.rerun()

            with col_fiyat:
                st.markdown(
                    f"<span style='font-size:13px'>{fiyat_gosterim}</span>",
                    unsafe_allow_html=True
                )

            with col_durum:
                st.markdown(
                    f"<span style='color:{renk_kz};font-weight:bold'>{durum_gosterim}</span>",
                    unsafe_allow_html=True
                )



            # Grafik (TEK BLOK)
            if st.session_state.get("grafik_aktif_hisse") == h:

                df_graf = grafik_verisi_indir(h + ".IS")

                if not df_graf.empty:

                    if isinstance(df_graf.columns, pd.MultiIndex):
                        df_graf.columns = df_graf.columns.droplevel(1)

                    fig = go.Figure(
                        data=[
                            go.Candlestick(
                                x=df_graf.index,
                                open=df_graf["Open"],
                                high=df_graf["High"],
                                low=df_graf["Low"],
                                close=df_graf["Close"]
                            )
                        ]
                    )

                    fig.update_layout(
                        template="plotly_dark",
                        height=250,
                        margin=dict(
                            l=0,
                            r=0,
                            t=10,
                            b=0
                        )
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

        st.write("")

        if st.button(
            "🔄 Verileri Yenile",
            key="mob_global_yenile"
        ):
            st.cache_data.clear()
            st.rerun()
            
# --- 2. SEKME: HİSSE ANALİZ ---
with sekme2:
    with st.form(key="analiz_arama_formu", clear_on_submit=True):
        analiz_girdisi = st.text_input("Hisse Kodu Giriniz").upper().strip()
        analiz_tetiklendi = st.form_submit_button("🚀 Analiz Et")
        if analiz_tetiklendi and analiz_girdisi:
            st.session_state["analiz_edilen_hisse"] = analiz_girdisi
    
    hisse_kodu = st.session_state["analiz_edilen_hisse"]
    
    if hisse_kodu:
        sorgu_kodu = hisse_kodu if hisse_kodu.endswith(".IS") else hisse_kodu + ".IS"
        try:
            df = yf.download(sorgu_kodu, period="60d", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            
            if not df.empty:
                kapanis = df['Close'].squeeze()
                son_fiyat = kapanis.iloc[-1]
                hedef_fiyat, tahmin_serisi = mobil_tahmin_motoru(df)
                potansiyel = ((hedef_fiyat - son_fiyat) / son_fiyat) * 100
                hacim_onay = df['Volume'].squeeze().iloc[-1] > (df['Volume'].squeeze().rolling(10).mean().iloc[-1] * 0.8)
                
                # Sinyal Hesaplama
                df['RSI'] = ta.momentum.rsi(kapanis, window=14)
                macd = ta.trend.MACD(kapanis)
                son_rsi, son_m, son_ms = df['RSI'].iloc[-1], macd.macd().iloc[-1], macd.macd_signal().iloc[-1]
                
                if son_rsi < 40 or (son_m > son_ms and son_rsi < 55): sinyal_metni, sinyal_rengi = "🟢 GÜÇLÜ AL", "#2ECC71"
                elif son_rsi > 65 or (son_m < son_ms and son_rsi > 50): sinyal_metni, sinyal_rengi = "🔴 GÜÇLÜ SAT", "#E74C3C"
                else: sinyal_metni, sinyal_rengi = "🟡 TUT / NÖTR", "#F1C40F"
                
                anlz_col1, anlz_col2 = st.columns([1, 2])
                with anlz_col1:
                    st.markdown(f"""
                    <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px; border: 1px solid #2D2D2D;'>
                        <h3 style='color: white;'>{hisse_kodu} Raporu</h3>
                        <p>Fiyat: <b>{son_fiyat:,.2f} TL</b></p>
                        <p>Hacim Onayı: <b>{'✅' if hacim_onay else '❌'}</b></p>
                        <p>Sinyal: <b style='color: {sinyal_rengi};'>{sinyal_metni}</b></p>
                        <h4 style='color: #00F0FF;'>🚀 YZ Hedef: {hedef_fiyat:.2f} (%{potansiyel:+.2f})</h4>
                    </div>
                    """, unsafe_allow_html=True)
                with anlz_col2:
                    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='#121212')
                    ax.set_facecolor('#1E1E1E')

                    # --- HASSAS EKSEN AYARLARI ---
                    ax.yaxis.set_major_locator(MultipleLocator(0.1)) 
                    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

                    # --- ÇİZGİLER ---
                    ax.plot(range(30), kapanis.tail(30).values, color='#00F0FF', label="Gerçek")
                    ax.plot(range(29, 35), np.concatenate(([kapanis.iloc[-1]], tahmin_serisi)), 
                            color='#FF00FF', linestyle='--', linewidth=2, label="YZ Tahmin")

                    # --- GÖRSEL DÜZENLEMELER ---
                    ax.tick_params(colors='white', labelsize=9)
                    ax.grid(True, color='#2D2D2D', linestyle='--')
                    ax.legend(loc='upper left', fontsize=8, facecolor='#1E1E1E', labelcolor='white')
                    
                    # Eksen kenarlarını (spine) yok ederek daha modern bir görünüm verelim
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    
                    fig.tight_layout()
                    st.pyplot(fig)
                
        except: st.error("Veri çekilemedi.")
     
# --- 3. SEKME: MEGA RADAR ---
with sekme3:
    col1, col2 = st.columns(2)
    sadece_guclu = col1.checkbox("GÜÇLÜ AL Sinyali", value=True)
    hacim_filtresi = col2.checkbox("Hacim Onayı İstiyorum", value=False)
    
    if st.button("🚀 TARAMAYI BAŞLAT", key="mob_radar_start"):
        guncel_hisse_listesi = dinamik_bist_listesi_yukle()
        bulunanlar = []
        toplam = len(guncel_hisse_listesi)
        
        ilerleme_bari = st.progress(0)
        durum_alani = st.empty()
        sonuc_alani = st.empty()  
        
        for idx, h in enumerate(guncel_hisse_listesi):
            durum_alani.write(f"<span style='color:white; font-size:14px;'>Taranıyor: {h} ({idx+1}/{toplam})</span>", unsafe_allow_html=True)
            ilerleme_bari.progress((idx + 1) / toplam)
            
            try:
                df = yf.download(h + ".IS", period="40d", interval="1d", progress=False)
                if df is None or len(df) < 20: continue
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                
                kapanis, hacim = df['Close'].squeeze(), df['Volume'].squeeze()
                son_rsi = ta.momentum.rsi(kapanis, window=14).iloc[-1]
                macd_c = ta.trend.MACD(kapanis).macd().iloc[-1]
                macd_s = ta.trend.MACD(kapanis).macd_signal().iloc[-1]
                
                sinyal_var = (son_rsi < 42 and macd_c > macd_s) or (sadece_guclu == False and son_rsi < 30)
                
                hacim_ort = hacim.rolling(10).mean().iloc[-1]
                hacim_onayli = hacim.iloc[-1] > (hacim_ort * 0.8)
                
                if sinyal_var:
                    if not hacim_filtresi or hacim_onayli:
                        bulunanlar.append(h)
                        with sonuc_alani.container():
                            st.success(f"✅ {len(bulunanlar)} adet hisse bulundu:")
                            for hisse in bulunanlar:
                                st.markdown(f"🔹 **{hisse}**")
            except: 
                continue
        
        durum_alani.text("Tarama tamamlandı!")
        ilerleme_bari.empty()
        
        if not bulunanlar:
            st.warning("Seçili kriterlerde hisse bulunamadı.")

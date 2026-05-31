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
import matplotlib.pyplot as plt
import matplotlib

# Matplotlib ayarları
matplotlib.use('Agg') 
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# ==============================================================================
# STREAMLIT SAYFA AYARLARI
# ==============================================================================
st.set_page_config(page_title="Mobil Borsa", layout="wide", initial_sidebar_state="collapsed")

# 1. Butona tıklandığında çalışacak mobil uyumlu fonksiyon
def grafik_tetikle(hisse_kodu, su_an_aktif_mi):
    if su_an_aktif_mi:
        st.session_state["grafik_aktif_hisse"] = None
    else:
        st.session_state["grafik_aktif_hisse"] = hisse_kodu


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
    
# ==============================================================================
# 2. GÜVENLİ VERİ MOTORU (Borsa Kapalıyken Kapanışı Korur)
# ==============================================================================
@st.cache_data(ttl=3600)
def dinamik_bist_listesi_yukle():
    csv_yolu = "bist_hisseler.csv"
    if os.path.exists(csv_yolu):
        df = pd.read_csv(csv_yolu)
        return df["kod"].tolist()
    
    return ["A1CAP", "ADEL", "AGROT", "AKBNK", "ALARK", "ASELS", "THYAO"]

@st.cache_data(ttl=60)  
def guvenli_fiyat_yakala(sorgu_kodu):
    """Borsa açıkken 5dk'lık canlı veri, kapalıyken en son günlük kapanış fiyatını döndürür."""
    try:
        df = yf.download(sorgu_kodu, period="1d", interval="5m", progress=False)
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.droplevel(1)
            
        if df is not None and not df.empty:
            return float(df["Close"].squeeze().iloc[-1])
    except:
        pass
        
    try:
        df_yedek = yf.download(sorgu_kodu, period="5d", interval="1d", progress=False)
        if isinstance(df_yedek.columns, pd.MultiIndex): 
            df_yedek.columns = df_yedek.columns.droplevel(1)
            
        if df_yedek is not None and not df_yedek.empty:
            return float(df_yedek["Close"].squeeze().iloc[-1])
    except:
        pass
        
    return None

@st.cache_data(ttl=300) 
def grafik_verisi_indir(sorgu_kodu):
    return yf.download(sorgu_kodu, period="3mo", interval="1d", progress=False)

# ==============================================================================
# 3. YAPAY ZEKA TAHMİNİ
# ==============================================================================
def mobil_tahmin_motoru(df):
    if df is None or df.empty or len(df) < 5: return 0.0, np.zeros(5)
    try:
        data = df.tail(60).copy()
        data['gun'] = range(len(data))
        model = HuberRegressor(max_iter=1000).fit(data[['gun']], data['Close'].squeeze())
        gelecek = pd.DataFrame({'gun': range(len(data), len(data) + 5)})
        tahmin = model.predict(gelecek)
        return tahmin[-1], tahmin
    except: return 0.0, np.zeros(5)

st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #FFFFFF; }
    div[data-testid="stExpander"] { background-color: #1E1E1E; border: 1px solid #2D2D2D; border-radius: 10px; }
    div.stFormSubmitButton > button { background-color: #007BFF !important; color: white !important; width: 100% !important; }
    div.stButton > button { background-color: #007BFF !important; color: white !important; }
    
    /* 1. SADECE + / - BUTONLARINI MİNİ YAPMA (Bunları kodda 'primary' olarak ayarlayacağız) */
    button[kind="primary"] {
        width: 28px !important;
        height: 28px !important;
        min-width: 28px !important;
        min-height: 28px !important;
        padding: 0px !important;
        font-size: 14px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 6px !important;
        margin-top: 5px !important;
        background-color: #2D2D2D !important;
        border: 1px solid #444 !important;
        color: #00F0FF !important;
    }

    /* 2. ALT BUTONLAR (SİL & ANALİZ) GÖRÜNÜMÜ VE RENKLERİ */
    /* İlk sütundaki buton (SİL) */
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button[kind="secondary"] {
        background-color: #E74C3C !important;
        color: white !important;
        font-size: 13px !important;
        font-weight: bold !important;
        white-space: nowrap !important;
        padding: 8px 5px !important;
        width: 100% !important;
        border: none !important;
    }
    
    /* İkinci sütundaki buton (ANALİZ) */
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #121212 !important;
        font-size: 13px !important;
        font-weight: bold !important;
        white-space: nowrap !important;
        padding: 8px 5px !important;
        width: 100% !important;
        border: none !important;
    }

    /* 3. MOBİLDE SİL VE ANALİZ BUTONLARININ YAN YANA KALMASINI ZORLAMA */
    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) {
            flex-direction: row !important;
            flex-wrap: nowrap !important;
        }
        div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) > div[data-testid="column"] {
            width: 50% !important;
            min-width: 50% !important;
            flex-basis: 50% !important;
        }
    }

    div[data-testid="stPopover"] button {
        width: 35px !important; height: 26px !important; background-color: #2D2D2D !important;
        border: 1px solid #444444 !important; color: #00F0FF !important;
    }
    div[data-testid="stPopoverBody"] button {
        background: none !important; color: white !important; text-align: left !important; width: 100% !important;
    }
    div[data-testid="stPopoverBody"] button:hover { background-color: #007BFF !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🖥️ Borsa")
db = Veritabani()

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

# Sekmeler tanımlanıyor
aktif_sekme = st.tabs(["PORTFÖY & STOP", "HİSSE ANALİZ", "MEGA RADAR"])

st.markdown("""
    <style>
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

# --- 1. SEKME: PORTFÖY (WIDGET TABLE) ---
with aktif_sekme[0]:
    hisserler = db.listeyi_getir()

    with st.expander("➕ Hisse Ekle / Düzenle"):
        with st.form(key="hisse_ekleme_formu", clear_on_submit=True):
            yeni_hisse = st.text_input("Hisse Kodu").upper().strip()
            c1, c2 = st.columns(2)
            maliyet = c1.number_input("Maliyet", value=0.0, step=0.01)
            adet = c2.number_input("Adet", value=0, step=1)
            if st.form_submit_button("Kaydet") and yeni_hisse:
                db.hisse_ekle(yeni_hisse, maliyet, adet)
                st.rerun()

    if not hisserler:
        st.warning("Takip listesi boş.")
    else:
        # --- TAMAMEN SABİT BAŞLIKLAR ---
        st.markdown("""
            <div style="
                background-color: #121212;
                display: flex;
                justify-content: space-between;
                font-weight: bold;
                font-size: 12px;
                color: #888888;
                padding-right: 45px;
                padding-top: 10px;
                padding-bottom: 5px;
                margin-bottom: 0px;
            ">
                <span style="width:25%; text-align:left;">HİSSE/ADET</span>
                <span style="width:25%; text-align:center;">FİYAT/MLY</span>
                <span style="width:25%; text-align:center;">K/Z (TL)</span>
                <span style="width:25%; text-align:right;">DEĞİŞİM</span>
            </div>
            <hr style="margin:0 0 5px 0; border:0; border-top:1px solid #333;">
        """, unsafe_allow_html=True)

        # --- KAYDIRILABİLİR ALAN BAŞLANGICI ---
        st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)

        for h, maliyet, adet in hisserler:
            sorgu = h if h.endswith(".IS") else h + ".IS"
            canli_fiyat = guvenli_fiyat_yakala(sorgu)
            
            if canli_fiyat is not None:
                toplam_maliyet = maliyet * adet
                
                if maliyet > 0:
                    kz_tl = (canli_fiyat - maliyet) * adet
                    degisim_yuzde = ((canli_fiyat - maliyet) / maliyet) * 100
                else:
                    kz_tl = 0.0
                    degisim_yuzde = 0.0

                renk = "#2ECC71" if kz_tl >= 0 else "#E74C3C"
                col_veri, col_btn = st.columns([88, 12])
                
                with col_veri:
                    st.markdown(f"""
                        <div style="display:flex; justify-content:space-between; align-items:center; height:35px;">
                            <div style="width:25%; text-align:left;">
                                <div style="color:#00F0FF; font-weight:bold; font-size:14px;">{h}</div>
                                <div style="color:#666; font-size:11px;">{adet} Ad.</div>
                            </div>
                            <div style="width:25%; text-align:center;">
                                <div style="color:white; font-size:14px;">{canli_fiyat:.2f}</div>
                                <div style="color:#666; font-size:11px;">M:{maliyet:.2f}</div>
                            </div>
                            <div style="width:25%; text-align:center;">
                                <div style="color:{renk}; font-size:13px; font-weight:500;">{kz_tl:+,.2f}</div>
                                <div style="color:#888; font-size:11px;">({toplam_maliyet:,.2f} TL)</div>
                            </div>
                            <div style="width:25%; text-align:right; color:{renk}; font-weight:bold; font-size:13px;">
                                %{degisim_yuzde:+.2f}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                with col_btn:
                    is_active = st.session_state.get("grafik_aktif_hisse") == h
                    button_label = "➖" if is_active else "➕"
                    
                    st.button(
                        button_label, 
                        key=f"btn_graf_{h}", 
                        use_container_width=True,
                        on_click=grafik_tetikle,
                        args=(h, is_active))

                # --- `+` BASINCA AÇILAN YAPİ KREDİ MOBİL DETAY PANELİ ---
                if st.session_state.get("grafik_aktif_hisse") == h:
                    df_gr = grafik_verisi_indir(sorgu)
                    if not df_gr.empty:
                        if isinstance(df_gr.columns, pd.MultiIndex): 
                            df_gr.columns = df_gr.columns.droplevel(1)
                        
                        # Mobil Günlük Veriler Alınıyor
                        try:
                            gun_yuksek = float(df_gr['High'].squeeze().iloc[-1])
                            gun_dusuk = float(df_gr['Low'].squeeze().iloc[-1])
                        except:
                            gun_yuksek, gun_dusuk = canli_fiyat, canli_fiyat

                        st.markdown("<div style='background-color: #1A1A1A; padding: 12px; border-radius: 8px; margin: 5px 0;'>", unsafe_allow_html=True)
                        
                        detay_col1, detay_col2 = st.columns([35, 65])
                        
                        with detay_col1:
                            st.markdown(f"""
                                <div style="font-size: 13px; line-height: 1.8; color: #BBBBBB; padding-top: 5px;">
                                    <div style="display:flex; justify-content:space-between; border-bottom: 1px solid #2D2D2D; padding-bottom:2px;"><span>Alış:</span><b style="color:white;">{canli_fiyat:.2f}</b></div>
                                    <div style="display:flex; justify-content:space-between; border-bottom: 1px solid #2D2D2D; padding-top:2px; padding-bottom:2px;"><span>Satış:</span><b style="color:white;">{canli_fiyat:.2f}</b></div>
                                    <div style="display:flex; justify-content:space-between; border-bottom: 1px solid #2D2D2D; padding-top:2px; padding-bottom:2px;"><span>Yüksek:</span><b style="color:#2ECC71;">{gun_yuksek:.2f}</b></div>
                                    <div style="display:flex; justify-content:space-between; padding-top:2px;"><span>Düşük:</span><b style="color:#E74C3C;">{gun_dusuk:.2f}</b></div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        with detay_col2:
                            kapanis = df_gr['Close'].squeeze()
                            hedef_fiyat, tahmin_serisi = mobil_tahmin_motoru(df_gr)
                            
                            # Şık Mini Grafik Çizimi
                            fig, ax = plt.subplots(figsize=(6, 2.5), facecolor='#1A1A1A')
                            ax.set_facecolor('#1E1E1E')
                            ax.plot(range(15), kapanis.tail(15).values, color='#00F0FF', linewidth=1.5, label="Gerçek")
                            ax.plot(range(14, 20), np.concatenate(([kapanis.iloc[-1]], tahmin_serisi)), color='#FF00FF', linestyle='--', linewidth=1.5, label="Tahmin")
                            ax.tick_params(colors='white', labelsize=7)
                            ax.grid(True, color='#2D2D2D', linestyle='--')
                            for spine in ax.spines.values():
                                spine.set_visible(False)
                            fig.tight_layout()
                            st.pyplot(fig)
                        
                        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
                        btn_alt1, btn_alt2 = st.columns(2)
                        
                        with btn_alt1:
                            if st.button("🗑️ Listemden Çıkar", key=f"detay_sil_{h}", use_container_width=True):
                                db.hisse_sil(h)
                                st.session_state["grafik_aktif_hisse"] = None
                                st.toast(f"❌ {h} başarıyla silindi.")
                                st.rerun()
                                
                        with btn_alt2:
                            if st.button("📈 Teknik Analizi Aç", key=f"detay_analiz_{h}", use_container_width=True):
                                st.session_state["analiz_edilen_hisse"] = h
                                # Hisse Analiz sekmesine yönlendirme simülasyonu
                                st.toast(f"🚀 {h} Analiz Laboratuvarına Aktarılıyor...")
                                st.rerun()
                                
                        st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown('<hr style="margin:5px 0; border:0; border-top:1px solid #1A1A1A;">', unsafe_allow_html=True)
            else:
                st.error(f"⚠️ {h} için bağlantı hatası oluştu.")

        st.markdown('</div>', unsafe_allow_html=True)
        # --- KAYDIRILABİLİR ALAN BİTİŞİ ---

    if st.button("🔄 Verileri Yenile", key="global_refresh_btn"):
        st.cache_data.clear()
        st.rerun()

# --- 2. SEKME: HİSSE ANALİZ ---
with aktif_sekme[1]:
    st.subheader("🔍 Detaylı Hisse Analiz Laboratuvarı")
    with st.form(key="analiz_arama_formu", clear_on_submit=True):
        analiz_girdisi = st.text_input("Hisse Kodu Girin ve Enter'a Basın (Örn: THYAO)", value=st.session_state["analiz_edilen_hisse"]).upper().strip()
        analiz_tetiklendi = st.form_submit_button("🚀 Analiz Et")
        if analiz_tetiklendi and analiz_girdisi:
            st.session_state["analiz_edilen_hisse"] = analiz_girdisi
    
    hisse_kodu = st.session_state["analiz_edilen_hisse"]
    if hisse_kodu:
        sorgu_kodu = hisse_kodu if hisse_kodu.endswith(".IS") else hisse_kodu + ".IS"
        try:
            df = yf.download(sorgu_kodu, period="3mo", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): 
                df.columns = df.columns.droplevel(1)
            
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
                
                # Rapor Paneli
                st.markdown(f"""
                    <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px; border: 1px solid #2D2D2D; margin-bottom: 15px;'>
                        <h3 style='color: white; margin:0 0 10px 0;'>{hisse_kodu} Raporu</h3>
                        <span style='font-size:15px; color:white;'>Fiyat: <b>{son_fiyat:,.2f} TL</b> | Hacim Onayı: <b>{'✅' if hacim_onay else '❌'}</b> | Sinyal: <b style='color: {sinyal_rengi};'>{sinyal_metni}</b></span>
                        <h4 style='color: #00F0FF; margin:10px 0 0 0;'>🚀 YZ Hedef: {hedef_fiyat:.2f} (%{potansiyel:+.2f})</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                # Tam Ekran Mumlu Grafik Tasarımı
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    increasing_line_color='#2ECC71',
                    decreasing_line_color='#E74C3C'
                )])
                
                fig.update_layout(
                    template="plotly_dark",
                    height=450,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_rangeslider_visible=True
                )
                st.plotly_chart(fig, use_container_width=True)
        except: 
            st.error("Veri çekilemedi.")
     
# --- 3. SEKME: MEGA RADAR ---
with aktif_sekme[2]:
    st.subheader("🔍 Radar Taraması")
    st.markdown("""
        <style>
        div[data-testid="stCheckbox"] label, div[data-testid="stCheckbox"] p { color: #FFFFFF !important; }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    sadece_guclu = col1.checkbox("Sadece GÜÇLÜ AL Sinyalleri", value=True)
    hacim_filtresi = col2.checkbox("Hacim Onayı İstiyorum", value=False)
   
    if st.button("🚀 TARAMAYI BAŞLAT", key="mob_radar_start"):
        guncel_hisse_listesi = dinamik_bist_listesi_yukle()
        bulunanlar = []
        toplam = len(guncel_hisse_listesi)
        
        ilerleme_bari = st.progress(0)
        durum_alani = st.empty()
        sonuc_alani = st.empty() 
        
        for idx, h in enumerate(guncel_hisse_listesi):
            durum_alani.write(f"<span style='color:white;'>Taranıyor: {h} ({idx+1}/{toplam})</span>", unsafe_allow_html=True)
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
                
                if sinyal_var and (not hacim_filtresi or hacim_onayli):
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

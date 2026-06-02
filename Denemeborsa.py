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
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib

from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer



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
    varsayilan_liste = ["A1CAP", "ADEL", "AGROT", "AKBNK", "ALARK", "ASELS", "THYAO"]
    csv_yolu = "bist_hisseler.csv"

    # Dosya yoksa doğrudan varsayılanı dön
    if not os.path.exists(csv_yolu):
        return varsayilan_liste
    try:
        df = pd.read_csv(csv_yolu)
        if df.empty or len(df.columns) == 0:
            return varsayilan_liste
        # 'kod' sütunu yoksa, belki ilk sütunu otomatik seçmek istersiniz
        sutun_adi = "kod" if "kod" in df.columns else df.columns[0]
        
        hisseler = (
            df[sutun_adi]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .unique()
            .tolist() )

        return hisseler if hisseler else varsayilan_liste

    except Exception as e:
        # Hata durumunu konsolda görmek geliştirme aşamasında çok işinize yarar
        st.error(f"Hisse listesi yüklenirken hata oluştu: {e}")
        return varsayilan_liste

# --- HIZLANDIRICI ÖNBELLEK FONKSİYONLARI ---
# --- Güncel Veri İndirme ---
@st.cache_data(ttl=60)
def guncel_fiyat_indir(sorgu_kodu):
    try:
        df = yf.download(sorgu_kodu, period="1d", interval="5m", progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=300) 
def grafik_verisi_indir(sorgu_kodu):
    return yf.download(sorgu_kodu, period="3mo", interval="1d", progress=False)

# --- 1. GÜVENLİ VERİ ÇEKME MOTORU ---
@st.cache_data(ttl=60)
def guvenli_fiyat_yakala(sorgu_kodu):
    try:
        df = yf.download(sorgu_kodu, period="5d", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if "Close" not in df.columns: return None
        close_seri = df["Close"].dropna()
        return float(close_seri.iloc[-1]) if not close_seri.empty else None
    except Exception: return None
# ==============================================================================
# 3. YAPAY ZEKA TAHMİN MOTORU
# ==============================================================================
def mobil_tahmin_motoru(df):

    try:
        # Veri kontrolü
        if df is None or df.empty or len(df) < 250:
            son = float(df["Close"].iloc[-1]) if df is not None and not df.empty else 0.0
            return {"son_tahmin": son, "seri": np.full(5, son), "alt": np.full(5, son), "ust": np.full(5, son), "hata_payi": 0.0}

        data = df.copy().tail(300)

        # Teknik göstergeler
        data["RSI"] = ta.momentum.RSIIndicator(close=data["Close"], window=14).rsi()
        data["MACD"] = ta.trend.MACD(close=data["Close"]).macd()
        data["EMA20"] = ta.trend.EMAIndicator(close=data["Close"], window=20).ema_indicator()
        data["EMA50"] = ta.trend.EMAIndicator(close=data["Close"], window=50).ema_indicator()
        data["EMA200"] = ta.trend.EMAIndicator(close=data["Close"], window=200).ema_indicator()
        data["ATR"] = ta.volatility.AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"]).average_true_range()
        
        bb = ta.volatility.BollingerBands(close=data["Close"], window=20, window_dev=2)
        data["BB_High"] = bb.bollinger_hband()
        data["BB_Low"] = bb.bollinger_lband()
        data["ADX"] = ta.trend.ADXIndicator(high=data["High"], low=data["Low"], close=data["Close"]).adx()

        # Lag ve Getiriler
        data["Close_1"] = data["Close"].shift(1)
        data["Close_2"] = data["Close"].shift(2)
        data["Close_3"] = data["Close"].shift(3)
        data["Close_5"] = data["Close"].shift(5)
        data["Return_1"] = data["Close"].pct_change(1)
        data["Return_5"] = data["Close"].pct_change(5)
        data["Return_20"] = data["Close"].pct_change(20)

        data["Target"] = data["Close"].shift(-1)
        data.dropna(inplace=True)

        ozellikler = ["Close", "Volume", "RSI", "MACD", "EMA20", "EMA50", "EMA200", "ATR", 
                      "BB_High", "BB_Low", "ADX", "Close_1", "Close_2", "Close_3", "Close_5", 
                      "Return_1", "Return_5", "Return_20"]

        X = data[ozellikler]
        y = data["Target"]

        imp = SimpleImputer(strategy="median")
        X_fit = imp.fit_transform(X)

        model = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)
        model.fit(X_fit, y)

        # Eğitim hatası hesapla
        train_pred = model.predict(X_fit)
        model_hatasi = np.std(y - train_pred)

        # Recursive Tahmin Döngüsü
        son_veri = data.iloc[-1]
        input_data = np.array([[son_veri[col] for col in ozellikler]])
        current_X = imp.transform(input_data)
        
        tahminler = []
        for _ in range(5):
            yeni_tahmin = model.predict(current_X)[0]
            tahminler.append(yeni_tahmin)
            # Bir sonraki gün için 'Close' değerini güncelle
            current_X[0, 0] = yeni_tahmin 
        
        seri = np.array(tahminler)

        # Hata payı ve Aralıklar
        volatilite = data["Close"].pct_change().rolling(20).std().iloc[-1]
        volatilite_hatasi = son_veri["Close"] * volatilite
        toplam_hata = (model_hatasi + volatilite_hatasi) / 2

        return {
            "son_tahmin": float(seri[-1]),
            "seri": seri,
            "alt": seri - (toplam_hata * 2),
            "ust": seri + (toplam_hata * 2),
            "hata_payi": float(toplam_hata)
        }

    except Exception:
        son = float(df["Close"].iloc[-1]) if df is not None and not df.empty else 0.0
        return {"son_tahmin": son, "seri": np.full(5, son), "alt": np.full(5, son), "ust": np.full(5, son), "hata_payi": 0.0}

# --- CSS PANEL ---
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #FFFFFF; }
    div[data-testid="stExpander"] { background-color: #1E1E1E; border: 1px solid #2D2D2D; border-radius: 10px; }
    div.stFormSubmitButton > button { background-color: #007BFF !important; color: white !important; width: 100% !important; }
    div.stButton > button { background-color: #007BFF !important; color: white !important; }
    
    /* PORTFÖYDEKİ + / - BUTONLARINI MİNİCİK YAPMA (YARI BOYUT) */
    div[data-testid="stHorizontalBlock"] div.stButton > button {
        width: 16px !important;
        height: 16px !important;
        min-width: 16px !important;
        min-height: 16px !important;
        padding: 0px !important;
        font-size: 9px !important;
        line-height: 16px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 4px !important;
        margin-top: 10px !important;
        background-color: #2D2D2D !important;
        border: none !important;
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
#Küçük bir zaman damgası ekleyerek kullanıcının sayfanın en son ne zaman yenilendiğini görmesini sağlıyoruz
st.caption(f"⏱️ Canlı takip tablosu 60 saniyede bir otomatik güncellenir. Son Yenilenme: {pd.Timestamp.now().strftime('%H:%M:%S')}")

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

sekme1, sekme2, sekme3 = st.tabs(["PORTFÖY & STOP", "HİSSE ANALİZ", "MEGA RADAR"])

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
with sekme1:
    
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
        
        st.markdown("""
            <style>
            .scrollable-container {
                max-height: 400px; /* Listenin kaplayacağı maksimum yükseklik */
                overflow-y: auto;  /* Veri sığmazsa kaydırma çubuğu çıkar */
                padding-right: 5px;
            }
            </style>
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
                
                
                # Saat bilgisini al
                su_an = datetime.now().strftime("%H:%M")
                
                with col_veri:
                    

                    st.markdown(f"""
                    <table style="width:100%; border:none; border-collapse: collapse; font-family: sans-serif;">
                        <tr>
                            <td style="width:25%; text-align:left; vertical-align:top; padding:0;">
                                <div style="color:#00F0FF; font-weight:bold; font-size:14px;">{h}</div>
                                <div style="color:#666; font-size:11px;">{adet} Ad.</div>
                            </td>
                            <td style="width:25%; text-align:center; vertical-align:top; padding:0;">
                                <div style="color:white; font-size:14px;">{canli_fiyat:.2f}</div>
                                <div style="color:#666; font-size:11px;">M:{maliyet:.2f}</div>
                            </td>
                            <td style="width:25%; text-align:center; vertical-align:top; padding:0;">
                                <div style="color:{renk}; font-size:13px; font-weight:500;">{kz_tl:+,.2f}</div>
                                <div style="color:#888; font-size:11px;">({toplam_maliyet:,.2f})</div>
                            </td>
                            <td style="width:25%; text-align:right; vertical-align:top; padding:0;">
                                <div style="color:{renk}; font-weight:bold; font-size:13px;">%{degisim_yuzde:+.2f}</div>
                            </td>
                        </tr>
                        <tr>
                            <td colspan="4" style="text-align:right; font-size:10px; color:#444; padding-top:5px; padding-bottom:10px;">
                                {su_an}
                            </td>
                        </tr>
                    </table>
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
                        
                        detay_col1, detay_col2 = st.columns([40, 60])
                        
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
                            # Bununla DEĞİŞTİRİN
                            tahmin_sonuc = mobil_tahmin_motoru(df_gr)

                            # Sözlük içindeki değerleri değişkenlere atayalım
                            hedef_fiyat = tahmin_sonuc["son_tahmin"]
                            tahmin_serisi = tahmin_sonuc["seri"]
                            alt_sinir = tahmin_sonuc["alt"]
                            ust_sinir = tahmin_sonuc["ust"]

                            
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
                        
                        # Butonlar için mobil uyumlu ve esnek yan yana yerleşim düzeni
                        btn_alt1, btn_alt2 = st.columns([45, 55])
                        
                        with btn_alt1:
                            if st.button("🗑️ Sil", key=f"detay_sil_{h}", use_container_width=True):
                                db.hisse_sil(h)
                                st.session_state["grafik_aktif_hisse"] = None
                                st.toast(f"❌ {h} başarıyla silindi.")
                                st.rerun()
                                
                        with btn_alt2:
                            if st.button("📈 Teknik Analiz", key=f"detay_analiz_{h}", use_container_width=True):
                                st.session_state["analiz_edilen_hisse"] = h
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
        
        
        
        
        
        

# --- 2. SEKME: HİSSE ANALİZ (GÜNCELLENMİŞ VERSİYON) ---
with sekme2:
    st.subheader("🔍 Detaylı Hisse Analiz Laboratuvarı")
    with st.form(key="analiz_arama_formu", clear_on_submit=True):
        analiz_girdisi = st.text_input("Hisse Kodu Girin (Örn: THYAO)").upper().strip()
        analiz_tetiklendi = st.form_submit_button("🚀 Analiz Et")
        if analiz_tetiklendi and analiz_girdisi:
            st.session_state["analiz_edilen_hisse"] = analiz_girdisi
    
    hisse_kodu = st.session_state.get("analiz_edilen_hisse", "")
    if hisse_kodu:
        sorgu_kodu = hisse_kodu if hisse_kodu.endswith(".IS") else hisse_kodu + ".IS"
        try:
            df = yf.download(sorgu_kodu, period="300d", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): 
                df.columns = df.columns.droplevel(1)
            
            if not df.empty:
                kapanis = df['Close'].squeeze()
                son_fiyat = kapanis.iloc[-1]
                
                # TAHMİN MOTORUNU ÇAĞIRMA
                tahmin_sonuc = mobil_tahmin_motoru(df)
                
                hedef_fiyat = tahmin_sonuc["son_tahmin"]
                tahmin_serisi = tahmin_sonuc["seri"]
                alt_sinir = tahmin_sonuc["alt"]
                ust_sinir = tahmin_sonuc["ust"]
                
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
                    
                    # 1. Gerçek veriyi çiz
                    ax.plot(range(30), kapanis.tail(30).values, color='#00F0FF', label="Gerçek")
                    
                    # 2. X eksenini 6 nokta olacak şekilde güncelle: 29, 30, 31, 32, 33, 34
                    tahmin_x = range(29, 35) 
                    
                    # 3. Y verilerini birleştir
                    tahmin_y = np.concatenate(([son_fiyat], tahmin_serisi))
                    
                    # 4. Tahmini çiz
                    ax.plot(tahmin_x, tahmin_y, color='#FF00FF', linestyle='--', label="Tahmin")
                    
                    # 5. Güven aralığını çiz (Burası da 6 nokta olmalı)
                    ax.fill_between(tahmin_x, 
                                    np.concatenate(([son_fiyat], alt_sinir)), 
                                    np.concatenate(([son_fiyat], ust_sinir)), 
                                    color='#FF00FF', alpha=0.1, label="Güven Aralığı")
                    
                    ax.tick_params(colors='white')
                    ax.grid(True, color='#2D2D2D')
                    ax.legend(loc='upper left')
                    st.pyplot(fig)
            else:
                st.warning("Hisse verisi boş döndü.")
        except Exception as e:
            st.error(f"Analiz sırasında hata oluştu: {e}")
     
# --- 3. SEKME: MEGA RADAR ---
with sekme3:
    st.subheader("🔍 Radar Taraması")
    # --- SMART CSS PANEL EKLEMESİ ---
    st.markdown("""
        <style>
        /* Checkbox (Onay Kutusu) yazılarını beyaz yapar */
        div[data-testid="stCheckbox"] label, div[data-testid="stCheckbox"] p {
            color: #FFFFFF !important;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    sadece_guclu = col1.checkbox("Sadece GÜÇLÜ AL Sinyalleri", value=True)
    hacim_filtresi = col2.checkbox("Hacim Onayı İstiyorum", value=False)
   
    if st.button("🚀 TARAMAYI BAŞLAT", key="mob_radar_start"):
        guncel_hisse_listesi = dinamik_bist_listesi_yukle()
        bulunanlar = []
        toplam = len(guncel_hisse_listesi)
        
        # İlerleme elemanları
        ilerleme_bari = st.progress(0)
        durum_alani = st.empty()
        sonuc_alani = st.empty()  # Canlı sonuçlar için boş alan
        
        for idx, h in enumerate(guncel_hisse_listesi):
            
            # Taranıyor yazısı için:
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
                
                # Sinyal Kontrolleri
                sinyal_var = (son_rsi < 42 and macd_c > macd_s) or (sadece_guclu == False and son_rsi < 30)
                
                # Hacim Kontrolü
                hacim_ort = hacim.rolling(10).mean().iloc[-1]
                hacim_onayli = hacim.iloc[-1] > (hacim_ort * 0.8)
                
                if sinyal_var:
                    if not hacim_filtresi or hacim_onayli:
                        bulunanlar.append(h)
                        # Canlı Güncelleme: Her yeni bulunan hisseyi anında ve bir arada ekrana yazar
                        with sonuc_alani.container():
                            st.success(f"✅ {len(bulunanlar)} adet hisse bulundu:")
                            for hisse in bulunanlar:
                                st.markdown(f"🔹 **{hisse}**")
            except: 
                continue
        
        # İşlem bittiğinde
        durum_alani.text("Tarama tamamlandı!")
        ilerleme_bari.empty()
        
        if not bulunanlar:
            st.warning("Seçili kriterlerde hisse bulunamadı.")

# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:42:45 2026

@author: EmirAysu
"""

import os
import sys
import logging
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from matplotlib.ticker import MultipleLocator
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

# PyInstaller çevre değişkeni ayarı (Qt çakışmalarını önler)
if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
    qt_plugin_path = os.path.join(base_dir, "PyQt5", "Qt5", "plugins")
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_plugin_path

# Matplotlib arkada harici pencere açmasını engeller ve logları kapatır
import matplotlib
matplotlib.use('Agg') 
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# STANDART KÜTÜPHANELER
import sqlite3
import subprocess
import threading
import socket

# VERİ ANALİZİ VE GRAFİK KÜTÜPHANELERİ

import matplotlib.pyplot as plt
import yfinance as yf
import ta


# PYQT5 GÖRSEL BİLEŞENLERİ
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt

# --- DİNAMİK MOD TESPİTİ (Streamlit mi, PyQt5 mi?) ---
try:
    from streamlit.runtime import exists
    IS_STREAMLIT = exists()
except ImportError:
    IS_STREAMLIT = False
    
# Butona tıklandığında çalışacak mobil uyumlu fonksiyon
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

    def hisse_detay_getir(self, kod):
        self.cursor.execute("SELECT maliyet, adet FROM watchlist WHERE hisse_kodu = ?", (kod,))
        return self.cursor.fetchone()
    
# ==============================================================================
# 2. DİNAMİK BIST LİSTESİ MOTORU
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


TUM_BIST = dinamik_bist_listesi_yukle()

# ==============================================================================
# BASİT YAPAY ZEKA TAHMİN BOTU
# ==============================================================================

def create_features(df):
   

    close = df["Close"]

    df["rsi"] = ta.momentum.RSIIndicator(close, 14).rsi()
    df["macd"] = ta.trend.MACD(close).macd()
    df["macd_signal"] = ta.trend.MACD(close).macd_signal()

    df["ma50"] = close.rolling(50).mean()
    df["ma200"] = close.rolling(200).mean()

    df["return"] = close.pct_change()
    df["future_return"] = close.shift(-5) / close - 1  # 5 gün sonrası

    df = df.dropna()

    return df


def create_labels(df):
    df["target"] = (df["future_return"] > 0.02).astype(int)  # %2 üstü = BUY

    return df


from sklearn.ensemble import RandomForestClassifier


def train_model(df):
    features = ["rsi", "macd", "macd_signal", "ma50", "ma200", "return"]

    X = df[features]
    y = df["target"]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X, y)

    return model, features


def generate_signal(model, features, latest_row):
    X = latest_row[features].values.reshape(1, -1)

    prob = model.predict_proba(X)[0][1]

    if prob > 0.6:
        return "BUY", prob
    else:
        return "NO TRADE", prob


def risk_filter(rsi, prob):
    if rsi > 70:
        return False

    if prob < 0.6:
        return False

    return True


def real_ai_trading_system(hisseler):
   
    results = []

    for h in hisseler:
        df = yf.download(h + ".IS", period="400d", interval="1d", progress=False)

        if df is None or df.empty:
            continue

        df = create_features(df)
        df = create_labels(df)

        model, features = train_model(df)

        latest = df.iloc[-1]

        signal, prob = generate_signal(model, features, latest)

        if signal != "BUY":
            continue

        rsi = latest["rsi"]

        if not risk_filter(rsi, prob):
            continue

        price = latest["Close"]

        atr = ta.volatility.AverageTrueRange(
            df["High"], df["Low"], df["Close"], 14
        ).average_true_range().iloc[-1]

        sl = price - atr * 1.5
        tp = price + atr * 3

        results.append({
            "hisse": h,
            "probability": prob,
            "entry": price,
            "sl": sl,
            "tp": tp
        })

    return sorted(results, key=lambda x: x["probability"], reverse=True)

 

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

# ==============================================================================
# PYQT5 MASAÜSTÜ PENCERE SINIFI
# ==============================================================================
class BorsaMobilUygulama(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Borsa Kontrol Paneli")
        self.setGeometry(200, 200, 450, 220)
        self.setStyleSheet("background-color: #121212; color: white;")
        
        merkezi_widget = QWidget()
        layout = QVBoxLayout()
        
        self.baslik = QLabel("🖥️ Borsa Hibrit Motoru Aktif")
        self.baslik.setAlignment(Qt.AlignCenter)
        self.baslik.setStyleSheet("font-size: 18px; font-weight: bold; color: #00F0FF; margin-top: 10px;")
        
        self.acıklama = QLabel("Uygulama arka planda hazır.\nGelişmiş grafik ve analiz paneline erişmek için\naşağıdaki butona tıklayarak tarayıcı modunu başlatın.")
        self.acıklama.setAlignment(Qt.AlignCenter)
        self.acıklama.setStyleSheet("font-size: 12px; color: #aaaaaa; margin: 15px 0;")
        
        self.buton = QPushButton("🌐 Gelişmiş Web Panelini Başlat")
        self.buton.setStyleSheet("""
            QPushButton {
                background-color: #007BFF; 
                color: white; 
                padding: 12px; 
                font-weight: bold; 
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        self.buton.clicked.connect(self.streamlit_panel_ac)
        
        layout.addWidget(self.baslik)
        layout.addWidget(self.acıklama)
        layout.addWidget(self.buton)
        merkezi_widget.setLayout(layout)
        self.setCentralWidget(merkezi_widget)
        
    def streamlit_panel_ac(self):
        self.baslik.setText("🚀 Panel Başlatılıyor...")
        self.buton.setEnabled(False)
        subprocess.Popen(["streamlit", "run", sys.argv[0]])
        self.close()

# ==============================================================================
# 4. STREAMLIT MASAÜSTÜ UYGULAMA PANELİ
# ==============================================================================
if IS_STREAMLIT:    
    st.set_page_config(page_title="Masaüstü Borsa", layout="wide")
    
    # --- CSS PANEL ---
    st.markdown("""
        <style>
        .stApp { background-color: #121212; color: #FFFFFF; }
        div[data-testid="stExpander"] { background-color: #1E1E1E; border: 1px solid #2D2D2D; border-radius: 10px; }
        div.stFormSubmitButton > button { background-color: #007BFF !important; color: white !important; width: 100% !important; }
        
        /* Yenile ve Sil Butonlarının Ortak Mobil Uyumlu CSS Tasarımı */
        div.stButton > button { 
            background-color: #007BFF !important; 
            color: white !important;
            font-size: 12.5px !important; /* Mobil ekranda harflerin kırılmaması için font ideal boyuta getirildi */
            white-space: nowrap !important; /* Yazının kesinlikle aşağı satıra taşmasını engeller */
            padding: 8px 10px !important;
            font-weight: bold !important;
        }
        
        [data-testid="stMainBlockContainer"] {
            padding-top: 2rem !important;
        }
        
        /* PORTFÖYDEKİ KATMANLARIN VE KAPSAYICININ YÜKSEKLİĞİNİ SABİTLEME */
        .scrollable-container {
            max-height: 380px !important;
            overflow-y: auto !important;
            padding-right: 5px;
        }
        
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

        /* Checkbox hizalamasını düzenleme */
        div[data-testid="stCheckbox"] {
            margin-top: 8px !important;
        }

        /* Silme butonu için özel kırmızı stil */
        div.stButton > button[key^="global_delete_btn"] {
            background-color: #E74C3C !important;
            color: white !important;
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

    sekme1, sekme2, sekme3, sekme4 , sekme5= st.tabs(["PORTFÖY & STOP", "HİSSE ANALİZ", "MEGA RADAR","Mega 2","YAPAY ZEKA"])

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
            # Bu div sayesinde listeniz telefonda ekranı yukarı taşımayacak, kendi içinde kayacak.
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

                    if st.session_state.get("grafik_aktif_hisse") == h:
                        df_gr = grafik_verisi_indir(sorgu)
                        if not df_gr.empty:
                            if isinstance(df_gr.columns, pd.MultiIndex): 
                                df_gr.columns = df_gr.columns.droplevel(1)
                            fig = go.Figure(data=[go.Candlestick(x=df_gr.index, open=df_gr['Open'], high=df_gr['High'], low=df_gr['Low'], close=df_gr['Close'])])
                            fig.update_layout(template="plotly_dark", height=200, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
                            st.plotly_chart(fig, use_container_width=True)
                    
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
        st.subheader("📊 PRO TRADE v2 (Trade Management System)")

        col1, col2 = st.columns(2)
        risk_pct = col1.slider("İşlem başı risk %", 0.5, 5.0, 1.0)
        sadece_guclu = col2.checkbox("Sadece kaliteli trade (80+)", value=True)

        if st.button("🚀 PRO TRADE v2 BAŞLAT", key="pro_trade_v2"):

            

            hisseler = dinamik_bist_listesi_yukle()

            results = []
            max_trades_per_day = 5
            trade_count = 0

            progress = st.progress(0)
            status = st.empty()

            def get_data(symbol):
                try:
                    df = yf.download(symbol, period="300d", interval="1d", progress=False)
                    if df is None or df.empty:
                        return None
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)
                    return df.dropna()
                except:
                    return None

            for i, h in enumerate(hisseler):

                status.write(f"Taranıyor: {h}")
                progress.progress((i+1)/len(hisseler))

                if trade_count >= max_trades_per_day:
                    break

                df = get_data(h + ".IS")
                if df is None or len(df) < 200:
                    continue

                close = df["Close"].astype(float)
                high = df["High"].astype(float)
                low = df["Low"].astype(float)
                volume = df["Volume"].astype(float)

                try:
                    # ======================
                    # INDICATORS
                    # ======================
                    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]

                    ma50 = close.rolling(50).mean().iloc[-1]
                    ma200 = close.rolling(200).mean().iloc[-1]

                    trend_up = ma50 > ma200

                    vol_avg = volume.rolling(20).mean().iloc[-1]
                    vol_ok = volume.iloc[-1] > vol_avg * 1.2

                    rsi_ok = 35 < rsi < 50

                    entry_signal = trend_up and rsi_ok and vol_ok

                    if not entry_signal:
                        continue

                    # ======================
                    # PRICE + RISK MODEL
                    # ======================
                    price = close.iloc[-1]

                    atr = ta.volatility.AverageTrueRange(
                        high, low, close, window=14
                    ).average_true_range().iloc[-1]

                    stop_loss = price - (atr * 1.5)
                    take_profit = price + (atr * 3)

                    risk = price - stop_loss
                    reward = take_profit - price

                    rr = reward / risk if risk != 0 else 0

                    # ======================
                    # POSITION SIZE (RISK MODEL)
                    # ======================
                    capital = 10000  # varsayım
                    risk_amount = capital * (risk_pct / 100)

                    position_size = risk_amount / risk if risk != 0 else 0

                    # ======================
                    # TRADE QUALITY SCORE
                    # ======================
                    score = 0

                    if trend_up:
                        score += 30

                    if rsi_ok:
                        score += 20

                    if vol_ok:
                        score += 20

                    score += min(rr * 15, 25)

                    # ======================
                    # FILTERS
                    # ======================
                    if sadece_guclu and score < 80:
                        continue

                    if rr < 1.5:
                        continue

                    # ======================
                    # ADD TRADE
                    # ======================
                    results.append({
                        "hisse": h,
                        "price": price,
                        "sl": stop_loss,
                        "tp": take_profit,
                        "rr": rr,
                        "score": score,
                        "position_size": position_size
                    })

                    trade_count += 1

                except:
                    continue

            # ======================
            # SORT
            # ======================
            results = sorted(results, key=lambda x: x["score"], reverse=True)

            status.text("Tarama tamamlandı!")
            progress.empty()

            # ======================
            # OUTPUT
            # ======================
            if not results:
                st.warning("Uygun trade bulunamadı.")
            else:
                st.success(f"{len(results)} kaliteli trade bulundu")

                for r in results:
                    st.markdown(f"## 🔹 {r['hisse']} | Score: {r['score']}/100")

                    st.write(f"""
                    - 💰 Entry: {r['price']:.2f}
                    - 🛑 Stop Loss: {r['sl']:.2f}
                    - 🎯 Take Profit: {r['tp']:.2f}
                    - 📊 R/R: {r['rr']:.2f}
                    - 📦 Position Size: {r['position_size']:.2f} lot (simülasyon)
                    """)
                    
    with sekme4:
        st.subheader("📊 Sade Profesyonel Trade Sistemi")

        col1, col2 = st.columns(2)
        sadece_guclu = col1.checkbox("Sadece güçlü işlemler (R:R > 1.5)", value=True)
        hacim_filtresi = col2.checkbox("Hacim filtresi", value=True)

        if st.button("🚀 TRADE TARAMA BAŞLAT", key="trade_system"):

            import numpy as np
            import ta
            import yfinance as yf

            hisseler = dinamik_bist_listesi_yukle()
            results = []

            progress = st.progress(0)
            status = st.empty()

            def get_data(symbol):
                try:
                    df = yf.download(symbol, period="250d", interval="1d", progress=False)
                    if df is None or df.empty:
                        return None
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)
                    return df.dropna()
                except:
                    return None

            for i, h in enumerate(hisseler):

                status.write(f"Taranıyor: {h}")
                progress.progress((i+1)/len(hisseler))

                df = get_data(h + ".IS")
                if df is None or len(df) < 200:
                    continue

                close = df["Close"].astype(float)
                high = df["High"].astype(float)
                low = df["Low"].astype(float)
                volume = df["Volume"].astype(float)

                try:
                    # =========================
                    # INDICATORS
                    # =========================
                    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]

                    ma50 = close.rolling(50).mean().iloc[-1]
                    ma200 = close.rolling(200).mean().iloc[-1]

                    trend_up = ma50 > ma200

                    vol_avg = volume.rolling(20).mean().iloc[-1]
                    vol_ok = volume.iloc[-1] > vol_avg * 1.1

                    rsi_ok = 30 < rsi < 50

                    # =========================
                    # ENTRY CONDITION
                    # =========================
                    entry_signal = trend_up and rsi_ok

                    if not entry_signal:
                        continue

                    # =========================
                    # TRADE LEVELS
                    # =========================
                    last_price = close.iloc[-1]

                    atr = ta.volatility.AverageTrueRange(
                        high, low, close, window=14
                    ).average_true_range().iloc[-1]

                    stop_loss = last_price - (atr * 1.5)
                    take_profit = last_price + (atr * 3)

                    risk = last_price - stop_loss
                    reward = take_profit - last_price

                    rr_ratio = reward / risk if risk != 0 else 0

                    # =========================
                    # FILTER
                    # =========================
                    if hacim_filtresi and not vol_ok:
                        continue

                    if sadece_guclu and rr_ratio < 1.5:
                        continue

                    results.append((
                        h,
                        last_price,
                        stop_loss,
                        take_profit,
                        rr_ratio,
                        rsi,
                        trend_up
                    ))

                except:
                    continue

            # =========================
            # SORT
            # =========================
            results = sorted(results, key=lambda x: x[4], reverse=True)

            status.text("Tarama tamamlandı!")
            progress.empty()

            # =========================
            # OUTPUT
            # =========================
            if not results:
                st.warning("Uygun trade fırsatı bulunamadı.")
            else:
                st.success(f"{len(results)} trade fırsatı bulundu")

                for r in results:
                    h, price, sl, tp, rr, rsi, trend = r

                    st.markdown(f"## 🔹 {h}")

                    st.write(f"""
                    - 💰 Entry: {price:.2f}
                    - 🛑 Stop Loss: {sl:.2f}
                    - 🎯 Take Profit: {tp:.2f}
                    - 📊 R/R Ratio: {rr:.2f}
                    - RSI: {rsi:.2f}
                    - Trend: {'UP' if trend else 'DOWN'}
                    """)  
    with sekme5:
        st.subheader("🤖 REAL AI TRADING SYSTEM (Tam Piyasa Taraması)")
        st.markdown("<span style='color:#00F0FF; font-size:14px;'>Random Forest algoritması son 1 yıllık teknik veriyi öğrenerek bir sonraki günün yönünü tahmin eder. BIST'teki tüm hisseler taranacaktır.</span>", unsafe_allow_html=True)

        if st.button("🚀 AI MODELİNİ TÜM PİYASA İÇİN ÇALIŞTIR", key="run_ai_btn_full"):
            
            import yfinance as yf
            import pandas as pd
            import ta
            from sklearn.ensemble import RandomForestClassifier

            hisseler = dinamik_bist_listesi_yukle()
            
            if not hisseler:
                st.error("⚠️ Hisse listesi yüklenemedi!")
            else:
                st.warning(f"⚠️ DİKKAT: BIST'teki tüm hisseler ({len(hisseler)} adet) taranıyor. Bu işlem işlemci hızına bağlı olarak 10-15 dakika sürebilir. Lütfen işlem bitene kadar sekmeyi kapatmayın veya yenilemeyin!")
                
                # 1. Hızlı Veri İndirme (Toplu - Tüm Piyasa)
                semboller = [f"{h}.IS" for h in hisseler]
                toplu_veri = yf.download(semboller, period="1y", interval="1d", progress=False)

                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                # 2. Yapay Zeka Modeli Eğitimi ve Tahmin Döngüsü
                for i, h in enumerate(hisseler):
                    sembol = f"{h}.IS"
                    status_text.text(f"🧠 Model Eğitiliyor: {h} ({i+1}/{len(hisseler)})")
                    progress_bar.progress((i + 1) / len(hisseler))
                    
                    try:
                        # Toplu veriden tekil hisseyi ayıkla
                        if len(semboller) > 1:
                            df = toplu_veri.xs(sembol, level=1, axis=1).dropna()
                        else:
                            df = toplu_veri.dropna()
                            
                        if len(df) < 100:  # AI eğitimi için minimum veri şartı
                            continue

                        # --- FEATURE ENGINEERING (Özellik Çıkarımı) ---
                        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
                        macd = ta.trend.MACD(df['Close'])
                        df['MACD'] = macd.macd()
                        df['MACD_Diff'] = macd.macd_diff()
                        df['Return'] = df['Close'].pct_change()
                        
                        # Hedef Değişken (Target): Bir sonraki günün kapanışı bugünden yüksekse 1, değilse 0
                        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
                        df = df.dropna()
                        
                        # --- MAKİNE ÖĞRENMESİ (Eğitim ve Test) ---
                        X = df[['RSI', 'MACD', 'MACD_Diff', 'Return']].iloc[:-1]
                        y = df['Target'].iloc[:-1]
                        X_tahmin = df[['RSI', 'MACD', 'MACD_Diff', 'Return']].iloc[-1:]
                        
                        # Modeli kur ve eğit
                        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
                        model.fit(X, y)
                        
                        # Yarın için Yükseliş İhtimalini hesapla
                        yukselis_ihtimali = model.predict_proba(X_tahmin)[0][1]
                        
                        # Sadece kazanma ihtimali %65'ten büyük olanları filtrele
                        if yukselis_ihtimali > 0.65:
                            last_price = df['Close'].iloc[-1]
                            
                            atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range().iloc[-1]
                            stop_loss = last_price - (atr * 1.5)
                            take_profit = last_price + (atr * 3)
                            
                            results.append({
                                "Hisse": h,
                                "AI Yükseliş İhtimali": yukselis_ihtimali,
                                "Giriş (Entry)": last_price,
                                "Stop Loss": stop_loss,
                                "Take Profit": take_profit
                            })
                            
                    except KeyError:
                        continue
                    except Exception as e:
                        continue
                        
                status_text.text("✅ Tüm Piyasa Yapay Zeka Analizi Tamamlandı!")
                progress_bar.empty()

                # 3. Sonuçları Tabloya Bas
                if not results:
                    st.warning("⚠️ Yapay Zeka tüm piyasayı taradı ancak şu anki piyasa koşullarında yeterince güçlü bir 'AL' sinyali bulamadı.")
                else:
                    st.success(f"🤖 Yapay Zeka {len(hisseler)} hisse içinden {len(results)} adet yüksek ihtimalli fırsat yakaladı!")
                    
                    df_ai_sonuc = pd.DataFrame(results)
                    df_ai_sonuc = df_ai_sonuc.sort_values(by="AI Yükseliş İhtimali", ascending=False).reset_index(drop=True)
                    
                    st.dataframe(
                        df_ai_sonuc.style.format({
                            "AI Yükseliş İhtimali": "{:.1%}",
                            "Giriş (Entry)": "{:.2f} ₺",
                            "Stop Loss": "{:.2f} ₺",
                            "Take Profit": "{:.2f} ₺"
                        }).background_gradient(subset=["AI Yükseliş İhtimali"], cmap="Greens"),
                        use_container_width=True
                    )
            
            
# ==============================================================================
# ÇALIŞTIRMA
# ==============================================================================
if __name__ == "__main__":
    if not IS_STREAMLIT:
        app = QApplication(sys.argv)
        pencere = BorsaMobilUygulama()
        pencere.show()
        sys.exit(app.exec_())

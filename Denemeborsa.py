# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 20:41:01 2026

@author: EmirAysu
"""

import os
import sys
import logging
import sqlite3
from pathlib import Path  # Hata önleyici yeni kütüphane
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import ta
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ==============================================================================
# --- DİNAMİK DOSYA YOLU VE KLASÖR AYARLARI (KODUN EN BAŞINDA OLMALIDIR) ---
# ==============================================================================
# Kodun çalıştığı ana dizini (D:\borsa) otomatik tespit eder
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if not getattr(sys, 'frozen', False) else sys._MEIPASS
MODEL_PATH = os.path.join(BASE_DIR, "models", "ensemble.pkl")

# KRİTİK HATA DÜZELTMESİ: os.makedirs yerine pathlib.Path kullanarak ntpath çakışması önlendi
Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

MODEL_FEATURES = ["rsi","macd","macd_signal", "ma50", "ma200"]


# Matplotlib arkada harici pencere açmasını engeller ve logları kapatır
matplotlib.use('Agg') 
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# PyInstaller çevre değişkeni ayarı (Qt çakışmalarını önler)
if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
    qt_plugin_path = os.path.join(base_dir, "PyQt5", "Qt5", "plugins")
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_plugin_path



# --- DİNAMİK MOD TESPİTİ ---
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
# 2. DİNAMİK BIST LİSTESİ MOTORU VE VERİ İNDİRME
# ==============================================================================
@st.cache_data(ttl=3600)
def dinamik_bist_listesi_yukle():
    varsayilan_liste = ["A1CAP", "ADEL", "AGROT", "AKBNK", "ALARK", "ASELS", "THYAO"]
    csv_yolu = "bist_hisseler.csv"

    if not os.path.exists(csv_yolu):
        return varsayilan_liste
    try:
        df = pd.read_csv(csv_yolu)
        if df.empty or len(df.columns) == 0:
            return varsayilan_liste
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
        st.error(f"Hisse listesi yüklenirken hata oluştu: {e}")
        return varsayilan_liste

@st.cache_data(ttl=60)
def guncel_fiyat_indir(sorgu_kodu):
    try:
        df = yf.download(sorgu_kodu, period="1d", interval="5m", progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except Exception: return pd.DataFrame()
    
@st.cache_data(ttl=3600)
def hisse_verisi_indir(kod):
    try:
        veri = yf.download(kod, period="300d", interval="1d", progress=False)
        return veri
    except Exception:
        return pd.DataFrame()

def grafik_verisi_indir(sorgu, periyot="1mo", aralik="1d"):
    try:
        hisse = yf.Ticker(sorgu)
        df = hisse.history(period=periyot, interval=aralik)
        return df if not df.empty else pd.DataFrame()
    except Exception as e:
        print(f"Grafik verisi çekilirken hata oluştu ({sorgu}): {e}")
        return pd.DataFrame()

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
# 3. ÖZELLİK VE ETİKETLEME MOTORU
# ==============================================================================
def create_features(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    close = df["Close"]

    df["rsi"] = ta.momentum.RSIIndicator(close, 14).rsi()
    macd_ind = ta.trend.MACD(close)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"] = macd_ind.macd_diff()

    df["ma5"] = close.rolling(5).mean()
    df["ma10"] = close.rolling(10).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()
    df["ma200"] = close.rolling(200).mean()

    df["return"] = close.pct_change()
    df["log_ret"] = np.log(close / close.shift(1))
    df["volatility"] = df["return"].rolling(10).std()
    df["volume_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
    
    df["future_return"] = close.shift(-5) / close - 1
    return df

def create_labels(df):
    df = df.copy()
    df["target"] = (df["future_return"] > 0.02).astype(int)
    df = df.dropna()
    return df

def detect_regime(df):
    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx().iloc[-1]
    if adx < 18:
        return "range"
    elif adx > 25:
        return "trend"
    else:
        return "transition"

# ==============================================================================
# 4. ENSEMBLE MODEL EĞİTİM MOTORU
# ==============================================================================
def train_models(X, y):
    xgb_m = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric="logloss",
        random_state=42
    )

    lgbm_m = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )

    rf_m = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    xgb_m.fit(X, y)
    lgbm_m.fit(X, y)
    rf_m.fit(X, y)

    return xgb_m, lgbm_m, rf_m

def ensemble_predict(xgb_m, lgbm_m, rf_m, X):
    p1 = xgb_m.predict_proba(X)[:,1]
    p2 = lgbm_m.predict_proba(X)[:,1]
    p3 = rf_m.predict_proba(X)[:,1]
    return (p1 * 0.5 + p2 * 0.35 + p3 * 0.15)[0]

# --- GARANTİLİ MODEL YÜKLEYİCİ VE OTOMATİK EĞİTİCİ ---
def guvenli_model_yukle():
    features_list = MODEL_FEATURES

    try:
        # Klasörü garanti et
        Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

        st.write("MODEL_PATH =", MODEL_PATH)
        st.write("Dosya mevcut mu?", os.path.exists(MODEL_PATH))

        if os.path.exists(MODEL_PATH):
            st.write("Dosya boyutu:", os.path.getsize(MODEL_PATH), "byte")

        # --------------------------------------------------
        # KAYITLI MODEL VARSA YÜKLE
        # --------------------------------------------------
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
            model_verisi = joblib.load(MODEL_PATH)

            if isinstance(model_verisi, dict):
                if "xgb" in model_verisi and "lgbm" in model_verisi:
                    st.success("✅ Kayıtlı model yüklendi.")
                    return model_verisi, features_list

            st.warning("⚠️ Model dosyası bozuk veya eski formatta.")

        # --------------------------------------------------
        # MODEL YOKSA YENİDEN EĞİT
        # --------------------------------------------------
        st.warning("⚠️ Model bulunamadı veya bozuk. Yeni model oluşturuluyor...")

        taslak_df = yf.download(
            "THYAO.IS",
            period="400d",
            interval="1d",
            progress=False
        )

        if taslak_df.empty:
            raise ValueError("Yfinance üzerinden eğitim verisi indirilemedi.")

        taslak_df = create_features(taslak_df)
        taslak_df = create_labels(taslak_df)

        X = taslak_df[features_list]
        y = taslak_df["target"]

        xgb_m, lgbm_m, rf_m = train_models(X, y)

        saved_model = {
            "xgb": xgb_m,
            "lgbm": lgbm_m,
            "rf": rf_m
        }

        joblib.dump(saved_model, MODEL_PATH)
        st.success("✅ Model başarıyla eğitildi ve kaydedildi.")
        st.write("Kaydedildi mi?", os.path.exists(MODEL_PATH))

        return saved_model, features_list

    except Exception as e:
        st.error(f"❌ Model otomatik yapılandırılırken kritik hata oluştu: {e}")
        return None, features_list

# ==============================================================================
# 5. RADAR SİSTEMİ
# ==============================================================================
def generate_signal(model_dict, features, latest_row):
    X = latest_row[features].values.reshape(1, -1)
    prob = ensemble_predict(model_dict["xgb"], model_dict["lgbm"], model_dict["rf"], X)

    if prob > 0.6:
        return "BUY", prob
    else:
        return "NO TRADE", prob

def risk_filter(rsi, prob):
    if rsi > 70 or prob < 0.6:
        return False
    return True

def real_ai_trading_system(hisseler):
    results = []
    model_dict, features = guvenli_model_yukle()
    
    if model_dict is None:
        return results

    for h in hisseler:
        try:
            df = yf.download(h + ".IS", period="400d", interval="1d", progress=False)
            if df is None or df.empty: continue

            df_feat = create_features(df)
            if df_feat.empty: continue
            
            latest = df_feat.iloc[-1]
            if latest[features].isna().any(): continue

            signal, prob = generate_signal(model_dict, features, latest)

            if signal != "BUY": continue
            rsi = float(latest["rsi"])
            if not risk_filter(rsi, prob): continue

            price = float(latest["Close"])
            
            atr_indicator = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], 14)
            atr = float(atr_indicator.average_true_range().dropna().iloc[-1])

            sl = price - atr * 1.5
            tp = price + atr * 3

            results.append({
                "hisse": h,
                "probability": float(prob),
                "entry": price,
                "sl": sl,
                "tp": tp
            })
        except Exception:
            continue

    return sorted(results, key=lambda x: x["probability"], reverse=True)

# ==============================================================================
# 6. MOBİL REGRESYON TAHMİN MOTORU
# ==============================================================================
def mobil_tahmin_motoru(df):
    try:
        if df is None or df.empty or len(df) < 250:
            son = float(df["Close"].iloc[-1]) if df is not None and not df.empty else 0.0
            return {"son_tahmin": son, "seri": np.full(5, son), "alt": np.full(5, son), "ust": np.full(5, son), "hata_payi": 0.0}

        data = df.copy().tail(300)

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

        train_pred = model.predict(X_fit)
        model_hatasi = np.std(y - train_pred)

        son_veri = data.iloc[-1]
        input_data = np.array([[son_veri[col] for col in ozellikler]])
        current_X = imp.transform(input_data)
        
        tahminler = []
        for _ in range(5):
            yeni_tahmin = model.predict(current_X)[0]
            tahminler.append(yeni_tahmin)
            current_X[0, 0] = yeni_tahmin 
        
        seri = np.array(tahminler)

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
        
        div.stButton > button { 
            background-color: #007BFF !important; 
            color: white !important;
            font-size: 12.5px !important; 
            white-space: nowrap !important; 
            padding: 8px 10px !important;
            font-weight: bold !important;
        }
        
        [data-testid="stMainBlockContainer"] {
            padding-top: 2rem !important;
        }
        
        .scrollable-container {
            max-height: 380px !important;
            overflow-y: auto !important;
            padding-right: 5px;
        }
        
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

        div[data-testid="stCheckbox"] {
            margin-top: 8px !important;
        }

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

    sekme1, sekme2, sekme3, sekme4, sekme5 = st.tabs(["PORTFÖY & STOP", "HİSSE ANALİZ", "MEGA RADAR","Mega 2","YAPAY ZEKA"])

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
            
            
            
            
 
    # --- 2. SEKME: HİSSE ANALİZ (GÜNCELLENMİŞ VE GÜVENLİ VERSİYON) ---
    with sekme2:
        st.subheader("🔍 Detaylı Hisse Analiz Laboratuvarı")

        # Arama Formu Bölümü
        with st.form(key="analiz_arama_formu", clear_on_submit=True):
            analiz_girdisi = st.text_input("Hisse Kodu Girin (Örn: THYAO)").upper().strip()
            analiz_tetiklendi = st.form_submit_button("🚀 Analiz Et")

            if analiz_tetiklendi and analiz_girdisi:
                st.session_state["analiz_edilen_hisse"] = analiz_girdisi
                
        hisse_kodu = st.session_state.get("analiz_edilen_hisse", "")

        if hisse_kodu:
            sorgu_kodu = hisse_kodu if hisse_kodu.endswith(".IS") else hisse_kodu + ".IS"

            try:
                # Önbellekten veriyi çekiyoruz
                df = hisse_verisi_indir(sorgu_kodu)

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
              
                if not df.empty:
                    # ==============================================================
                    # GÜVENLİ DEĞİŞKEN BAŞLATMA BLOĞU (Çökmeleri Önler)
                    # ==============================================================
                    puan = 0
                    guven_skoru = 0
                    smart_money = 0
                    radar_puan = 0
                    basari_olasiligi = 0
                    ai_yorum = "Analiz verisi yetersiz"
                    sinyal_metni, sinyal_rengi = "N/A", "#808080"
                    genel_not = 0
                    derece = "N/A"
                    trend_gucu = "🟡 Veri Yetersiz"
                    bb_sinyal = "🟡 Veri Yetersiz"
                    adx_yorum = "🟡 Veri Yetersiz"
                    hacim_yorum = "⚪ Normal"
                    son_rsi, son_m, son_ms, son_adx = 50, 0, 0, 0

                    # ==============================================================
                    # 1. MATEMATİKSEL VE TEKNİK HESAPLAMALAR BLOĞU
                    # ==============================================================
                    # DataFrame içindeki boş satırları temizle
                    df = df.dropna(subset=["Close", "High", "Low", "Volume"])
                    # =========================
                    # 📊 ARZ / TALEP (VOLUME BASED FLOW)
                    # =========================

                    df["price_change"] = df["Close"].diff()

                    df["buy_pressure"] = np.where(df["price_change"] > 0, df["Volume"], 0)
                    df["sell_pressure"] = np.where(df["price_change"] < 0, df["Volume"], 0)

                    buy_volume = df["buy_pressure"].sum()
                    sell_volume = df["sell_pressure"].sum()

                    total_flow = buy_volume + sell_volume

                    if total_flow > 0:
                        alim_orani = (buy_volume / total_flow) * 100
                        satim_orani = (sell_volume / total_flow) * 100
                    else:
                        alim_orani = 50
                        satim_orani = 50

                    # Smart Flow Index (ek feature gibi kullanabilirsin)
                    flow_index = alim_orani - satim_orani

                    if df.empty or len(df) < 60:
                        st.warning("⚠️ Analiz için yeterli temiz geçmiş veri bulunmuyor.")
                        st.stop()

                    kapanis = df["Close"].squeeze()
                    son_fiyat = float(kapanis.dropna().iloc[-1])
                 
                    # ==============================================================
                    # YZ TAHMİN MOTORU & MODEL EĞİTİM ENTEGRASYONU
                    # ==============================================================
                    try:
                        # Model özellikleri (features) oluşturuluyor
                        df_feat = create_features(df)
                        missing = [f for f in MODEL_FEATURES if f not in df_feat.columns]
                        if missing:
                            st.error(f"Eksik feature: {missing}")
                            st.stop()
                            
                        # Eğitim için X ve y hazırlanıyor (Geçmiş veriler)
                        features_list = MODEL_FEATURES
                        X_live = df_feat[features_list].iloc[-1:]
                                          
                        # Yapay hedef değişkeni oluşturma (Örn: 3 gün sonrası yukarıda mı?)
                        df_feat["future_return"] = df_feat["Close"].shift(-3) / df_feat["Close"] - 1
                        df_feat["target"] = np.where(df_feat["future_return"] > 0.01, 1, 0)
                        
                        X_train = df_feat[features_list].iloc[:-3]
                        y_train = df_feat["target"].iloc[:-3]
                        
                        # MODELLERİ ANLIK OLARAK EĞİTİYORUZ (Hafızada yoksa çökmesini önler)
                        models, _ = guvenli_model_yukle()

                        if models is None:
                            st.stop()

                        xgb_model = models["xgb"]
                        lgbm_model = models["lgbm"]
                                                                                                        
                        # Son günün verisiyle tahmin üretme
                        proba_xgb = xgb_model.predict_proba(X_live)[:, 1]
                        proba_lgbm = lgbm_model.predict_proba(X_live)[:, 1]

                        # İki modelin tahminlerinin ortalamasını alıyoruz
                        ham_proba = proba_xgb * 0.5 + proba_lgbm * 0.5

                        # Scalar / Array Uyumluluk Kontrolü (Hata Çözücü)
                        if hasattr(ham_proba, "__len__") and len(ham_proba) > 0:
                            proba = float(np.array(ham_proba).reshape(-1)[0])
                        else:
                            proba = float(ham_proba)
                        
                        # Proba değerine göre hedef fiyat simülasyonu
                        potansiyel = (proba - 0.5) * 20
                        potansiyel = np.clip(potansiyel, -5, 10)
                        
                        hedef_fiyat = son_fiyat * (1 + (potansiyel / 100))
                        
                        # Grafik için yapay bir tahmin serisi ve güven aralığı (5 Günlük Projeksiyon)
                        model_tahmini = np.array([
                            son_fiyat * (1 + (potansiyel / 100) * np.sqrt(i) / 5)
                            for i in range(1, 6)
                        ], dtype=float)
                        
                        ret = df["Close"].pct_change().dropna()
                        vol = ret.std() * son_fiyat

                        monte_carlo_sim = []
                        last = son_fiyat

                        for i in range(1, 6):
                            noise = np.random.normal(0, vol * 0.2)
                            last = last * (1 + (potansiyel / 100) / 5) + noise
                            monte_carlo_sim.append(last)

                        monte_carlo_sim = np.array(monte_carlo_sim, dtype=float)
                        monte_carlo_sim = np.maximum(monte_carlo_sim, 0.01)

                        tahmin_serisi = 0.7 * model_tahmini + 0.3 * monte_carlo_sim
                        tahmin_serisi = np.nan_to_num(tahmin_serisi, nan=son_fiyat)
                        
                        tahmin_serisi = np.clip(tahmin_serisi, son_fiyat * 0.7, son_fiyat * 1.5)
                        
                        alt_sinir = tahmin_serisi * 0.96
                        ust_sinir = tahmin_serisi * 1.04
                        
                    except Exception as model_err:
                        st.warning(
                            "⚠️ 'models/ensemble.pkl' bulunamadı veya bozuk! "
                            "Model sıfırdan eğitiliyor, lütfen bekleyin..."
                        )
                        st.error(f"❌ Model hatası: {model_err}")

                        proba = 0.5
                        hedef_fiyat = son_fiyat
                        potansiyel = 0.0
                        tahmin_serisi = None
                        alt_sinir = None
                        ust_sinir = None  

                    # fallback 
                    if tahmin_serisi is None or len(tahmin_serisi) == 0:
                        tahmin_serisi = np.array([son_fiyat] * 5)
                        alt_sinir = tahmin_serisi * 0.98
                        ust_sinir = tahmin_serisi * 1.02             
                    
                    # Hacim Onay Kontrolü
                    hacim_onay = len(df) >= 10 and df["Volume"].iloc[-1] > (df["Volume"].rolling(10).mean().iloc[-1] * 0.8)

                    # RSI & MACD Hesaplamaları
                    if len(df) >= 30:
                        df["RSI"] = ta.momentum.rsi(kapanis, window=14)
                        macd = ta.trend.MACD(kapanis)
                        rsi_series = df["RSI"].dropna()
                        son_rsi = float(rsi_series.iloc[-1]) if len(rsi_series) > 0 else 50
                        
                        son_m = float(macd.macd().dropna().iloc[-1])
                        son_ms = float(macd.macd_signal().dropna().iloc[-1])
                        
                    # ADX Analizi
                    if len(df) >= 20:
                        adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"])
                        son_adx = adx.adx().iloc[-1]
                        
                        if son_adx >= 30:
                            adx_yorum = "🔥 Çok Güçlü Trend"
                        elif son_adx >= 20:
                            adx_yorum = "🟢 Trend Var"
                        else:
                            adx_yorum = "⚠️ Zayıf Trend"
                                                                                                     
                    # =========================
                    # MA200 ve MA50 Trend Analizi
                    # =========================
                    ma200_kontrolu = len(df) >= 200
                    trend_confirmed = True  # default
                    if ma200_kontrolu:
                        ma200 = float(df["Close"].rolling(200).mean().iloc[-1])
                        ma50 = float(df["Close"].rolling(50).mean().iloc[-1]) if len(df) >= 50 else None
                        # -------------------------
                        # 1. TREND TANIMI
                        # -------------------------
                        if ma50 is not None and ma50 > ma200 and son_adx > 25:
                            trend = "🚀 Güçlü Yükseliş"
                        elif ma50 is not None and ma50 > ma200:
                            trend = "🟢 Yükseliş Eğilimi"
                        elif son_fiyat > ma200:
                            trend = "🟡 Zayıf Pozitif"
                        else:
                            trend = "🔴 Negatif"
                        trend_gucu = trend
                        # -------------------------
                        # 2. TREND VALIDATION (FILTER)
                        # -------------------------
                        if son_adx < 20 and son_fiyat < ma200:
                            trend_confirmed = False
                    else:
                        trend = "🟡 Yetersiz Veri (Yeni Arz)"
                        trend_gucu = "🟡 Veri Yetersiz"
                    
                    # Bollinger Bands Analizi
                    if len(df) >= 20:
                        bb = ta.volatility.BollingerBands(kapanis)
                        bb_ust = bb.bollinger_hband().iloc[-1]
                        bb_alt = bb.bollinger_lband().iloc[-1]
                        if son_fiyat < bb_alt:
                            bb_sinyal = "🟢 Aşırı Satım"
                        elif son_fiyat > bb_ust:
                            bb_sinyal = "🔴 Aşırı Alım"
                        else:
                            bb_sinyal = "🟡 Normal"
                                              
                                            
                    # Hacim Oranı Analizi 
                    ortalama_hacim = df["Volume"].rolling(20).mean().iloc[-1] if len(df) >= 20 else None
                        
                    if pd.notna(ortalama_hacim) and ortalama_hacim > 0:
                        hacim_orani = df["Volume"].iloc[-1] / ortalama_hacim
                    else:
                        hacim_orani = 1
                    
                    if hacim_orani > 2:
                        hacim_yorum = "🚀 Hacim Patlaması"
                    elif hacim_orani > 1.2:
                        hacim_yorum = "🟢 Güçlü Hacim"
                        
                    # Destek / Direnç Hesaplama
                    destek = float(df["Low"].tail(20).min()) if len(df) >= 20 else son_fiyat
                    direnc = float(df["High"].tail(20).max()) if len(df) >= 20 else son_fiyat
                        
                    # Risk / Getiri Oranı
                    risk = son_fiyat - destek
                    getiri = direnc - son_fiyat

                    rr = None
                    rr_text = ""
                    if risk <= 0:
                        rr = None
                        rr_text = "Fiyat destek altında / risk hesaplanamaz"
                    else:
                        rr = round(getiri / risk, 2)
                        rr_text = str(rr)

                    # Güven skoru başlangıç şartı (Numeric R/G kontrolü)
                    if isinstance(rr, (int, float)) and rr > 1.5:
                        guven_skoru += 10
                    
                    # ATR & Stop Loss / Kar Al               
                    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range().iloc[-1] if len(df) >= 14 else 0
                    stop_loss = son_fiyat - (atr * 1.5)
                    kar_al = son_fiyat + (atr * 3)

                    # ==============================================================
                    # 2. SKORLAMA VE SİNYAL MOTORLARI (REFACTOR EDİLMİŞ)
                    # ==============================================================
                    
                    # A. Güven Skoru Hesaplama
                    
                    momentum_score = 0
                    # RSI
                    if son_rsi < 30:
                        momentum_score += 25
                    elif son_rsi < 50:
                        momentum_score += 10

                    # MACD
                    if son_m > son_ms:
                        momentum_score += 25

                    # Volume
                    if hacim_onay:
                        momentum_score += 15

                    # Trend
                    if ma200_kontrolu and son_fiyat > ma200:
                        momentum_score += 20

                    # Risk/Reward
                    if isinstance(rr, (int, float)) and rr > 1.5:
                        momentum_score += 10
                    guven_skoru = min(95, momentum_score)

                    # B. Hisse Karnesi Puanlama
                    puan += 20 if son_m > son_ms else 0
                    if son_rsi < 35: puan += 25
                    elif son_rsi < 50: puan += 15
                    elif son_rsi < 60: puan += 5
                    puan += 20 if hacim_onay else -10
                    if ma200_kontrolu: puan += 20 if son_fiyat > ma200 else -15
                    puan = max(0, min(100, puan))

                    # C. Smart Money (TEMİZ MODEL - duplicate sinyal yok)
                    smart_money = 0
                    # Trend yönü (en önemli sinyal)
                    if son_m > son_ms:
                        smart_money += 20
                    # Hacim akışı (gerçek para girişi)
                    if hacim_orani > 1.5:
                        smart_money += 30
                    elif hacim_orani < 1:
                        smart_money -= 15
                    # Kurumsal trend filtresi (MA200)
                    if ma200_kontrolu:
                        if son_fiyat > ma200:
                            smart_money += 20
                        else:
                            smart_money -= 15
                    # Aşırı alım filtresi (smart exit behavior)
                    if son_rsi > 70:
                        smart_money -= 10
                    elif son_rsi < 50:
                        smart_money += 10
                    # Final clamp
                    smart_money = max(0, min(100, smart_money))
                    
                    # D. Radar Skoru Hesaplama
                    radar_puan = puan
                    if son_adx > 25: radar_puan += 10
                    if hacim_orani > 1.5: radar_puan += 10
                    if potansiyel > 10: radar_puan += 10
                    radar_puan = min(100, radar_puan)
                    
                    # E. Başarı Olasılığı Skoru
                    basari_olasiligi = 0

                    # Momentum
                    if son_m > son_ms:
                        basari_olasiligi += 25
                    # RSI (daha gerçekçi)
                    if son_rsi < 30:
                        basari_olasiligi += 20
                    elif son_rsi < 50:
                        basari_olasiligi += 10
                    # Hacim
                    if hacim_orani > 1.2:
                        basari_olasiligi += 15
                    elif hacim_orani < 1:
                        basari_olasiligi -= 5
                    # MA200
                    if ma200_kontrolu:
                        if son_fiyat > ma200:
                            basari_olasiligi += 20
                        else:
                            basari_olasiligi -= 15
                    # RR
                    if isinstance(rr, (int, float)):
                        if rr > 1.5:
                            basari_olasiligi += 10
                        elif rr < 1:
                            basari_olasiligi -= 10
                    # EK FİLTRELER
                    # 6. Trend Gücü (ADX) Kontrolü
                    if son_adx > 30:
                        basari_olasiligi += 10
                    elif son_adx < 20:
                        basari_olasiligi -= 15
                                            
                    if flow_index < -0.10:
                        basari_olasiligi -= 10
                    elif flow_index < -0.05:
                        basari_olasiligi -= 5
                        
                    if smart_money < 50:
                        basari_olasiligi -= 10
                    basari_olasiligi = max(0, min(100, basari_olasiligi))

                    # F. Yapay Zeka Yorum Motoru
                    if potansiyel > 10 and puan >= 75:
                        ai_yorum = "🚀 Güçlü yükseliş potansiyeli"
                    elif potansiyel > 3 and puan >= 60:
                        ai_yorum = "👍 Pozitif görünüm"
                    elif potansiyel > 0:
                        ai_yorum = "⚠️ Sınırlı yükseliş"
                    else:
                        ai_yorum = "🔴 Zayıf görünüm"

                    # G. Nihai Karar Sinyal Mekanizması (Paradoks Çözücü Katman)
                                   
                    # =========================
                    # VOLUME FLOW (CLEAN VERSION)
                    # =========================
                    # Price change
                    df["price_change"] = df["Close"].diff()
                    # Buy / Sell pressure
                    df["buy_pressure"] = np.where(df["price_change"] > 0, df["Volume"], 0)
                    df["sell_pressure"] = np.where(df["price_change"] < 0, df["Volume"], 0)

                    # Volume totals
                    buy_volume = df["buy_pressure"].sum()
                    sell_volume = df["sell_pressure"].sum()

                    # FLOW INDEX (NORMALIZED -1 to +1)
                    flow_index = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-9)               
                   
                    # =========================
                    # 2. MARKET REGIME
                    # =========================
                    volatilite = df["Close"].pct_change().rolling(10).std().iloc[-1] * 100
                    trend_market = (son_adx > 20 and ma200_kontrolu and ma200 and son_fiyat > ma200)
                    sideways_market = son_adx < 18 and volatilite < 1.5
                    distribution_market = (son_rsi > 70 and flow_index < 0)
                    accumulation_market = (son_rsi < 45 and flow_index > 0)
                    market_regime = "UNKNOWN"

                    if trend_market:
                        market_regime = "TREND"
                    elif sideways_market:
                        market_regime = "SIDEWAYS"
                    elif distribution_market:
                        market_regime = "DISTRIBUTION"
                    elif accumulation_market:
                        market_regime = "ACCUMULATION"

                    # =========================
                    # 4. NO TRADE FILTER
                    # =========================

                    trend_yetersiz = son_adx < 15
                    hacim_dusuk = (buy_volume + sell_volume) > 0 and (buy_volume + sell_volume) < df["Volume"].mean()
                    akis_stabil = abs(flow_index) < 0.03

                    asiri_alim = son_rsi > 75
                    trend_celiski = ma200_kontrolu and ma200 and son_fiyat < ma200 and son_rsi > 65

                    if (
                        (trend_yetersiz and hacim_dusuk and akis_stabil) or
                        asiri_alim or
                        trend_celiski
                    ):
                        sinyal_metni = "⚪ NO TRADE"
                        sinyal_rengi = "#808080"
                        no_trade_aktif = True

                    else:
                        no_trade_aktif = False

                        # =========================
                        # 6. FINAL SCORE ENGINE (FIXED & CLEAN)
                        # =========================
                        # FLOW SCORE (0-1 normalize)
                        flow_score = (flow_index + 1) / 2
                        flow_score = max(0, min(1, flow_score))
                        
                        # =========================
                        # BASE SCORE
                        # =========================
                        final_score = (
                            proba * 0.35 +
                            (guven_skoru / 100) * 0.20 +
                            (smart_money / 100) * 0.20 +
                            flow_score * 0.25
                        )
                        # =========================
                        # TREND / MARKET PENALTIES (TEK KATMAN)
                        # =========================
                        # Zayıf trend
                        if son_adx < 20:
                            final_score *= 0.85
                        # Çok zayıf trend (kritik)
                        if son_adx < 15:
                            final_score *= 0.75
                        # Negatif flow + zayıf trend
                        if son_adx < 20 and flow_index < 0:
                            final_score *= 0.92
                        # Trend onayı yoksa
                        if not trend_confirmed:
                            final_score *= 0.85
                        # =========================
                        # RISK PENALTIES
                        # =========================
                        # Düşük hacim
                        if hacim_orani < 0.7:
                            final_score *= 0.95
                        # Aşırı RSI
                        if son_rsi > 80:
                            final_score *= 0.90
                        # =========================
                        # FINAL CLAMP
                        # =========================
                        final_score = max(0, min(1, final_score))
                        
                        # SADECE RİSK DURUMUNU CEZALANDIR
                        if market_regime == "DISTRIBUTION":
                            final_score *= 0.90
                        elif market_regime == "SIDEWAYS":
                            final_score *= 0.95
                        final_score = max(0, min(1, final_score))
                        # =========================
                        # 7. REGIME MULTIPLIER
                        # =========================
                        regime_multiplier = 1.0

                        if market_regime == "TREND":
                            regime_multiplier = 1.10
                        elif market_regime == "SIDEWAYS":
                            regime_multiplier = 0.85
                        elif market_regime == "DISTRIBUTION":
                            regime_multiplier = 0.70
                        elif market_regime == "ACCUMULATION":
                            regime_multiplier = 1.05
                        elif market_regime == "UNKNOWN":
                            regime_multiplier = 0.80

                        final_score *= regime_multiplier

                        # =========================
                        # 9. SIGNAL
                        # =========================

                        # Sinyal Sınıflandırma
                        if son_adx < 20 and flow_index < -0.10:
                            sinyal_metni = "🟡 TUT / İZLE"
                            sinyal_rengi = "#FFC107"   # Sarı
                        elif final_score >= 0.80:
                            sinyal_metni = "🚀 ÇOK GÜÇLÜ AL"
                            sinyal_rengi = "#00C853"   # Koyu Yeşil
                        elif final_score >= 0.70:
                            sinyal_metni = "🟢 AL"
                            sinyal_rengi = "#4CAF50"   # Yeşil
                        elif final_score >= 0.65:
                            sinyal_metni = "🟡 İZLE"
                            sinyal_rengi = "#FFD600"   # Altın Sarısı
                        elif final_score >= 0.50:
                            sinyal_metni = "⚪ NÖTR"
                            sinyal_rengi = "#90A4AE"   # Gri
                        else:
                            sinyal_metni = "🔴 SAT"
                            sinyal_rengi = "#E53935"   # Kırmızı
                            
                    # H. Tahmin Gücü Sınıflandırması
                    if potansiyel > 15: tahmin_gucu = "🔥 Çok Güçlü"
                    elif potansiyel > 8: tahmin_gucu = "🚀 Güçlü"
                    elif potansiyel > 3: tahmin_gucu = "👍 Pozitif"
                    else: tahmin_gucu = "⚠️ Zayıf"

                    # I. Genel Derece Notu Hesaplama (No Trade Düzeltmeli)
                    genel_not = (
                        final_score * 100 * 0.50 +
                        puan * 0.20 +
                        guven_skoru * 0.15 +
                        smart_money * 0.15
                    )
                    genel_not = round(genel_not, 1)
                    st.write("Final Score:", round(final_score * 100, 2))
                    if no_trade_aktif:
                        genel_not = genel_not * 0.7  # Kullanıcıyı yanıltmamak için not %30 düşürülür
                        guven_skoru = min(guven_skoru, 50)  # Güven skoru baskılanır
                    
                    if genel_not >= 85: derece = "🏆 A+"
                    elif genel_not >= 75: derece = "🥇 A"
                    elif genel_not >= 65: derece = "🥈 B"
                    elif genel_not >= 50: derece = "🥉 C"
                    else: derece = "❌ D"
                    genel_not = round(genel_not, 1)

                    # ==============================================================
                    # 3. STREAMLIT GÖRSEL ÇIKTI PANELİ
                    # ==============================================================
                    st.success(ai_yorum)

                    # 3'lü Bloklar Halinde KPI Panelleri
                    st.markdown("## 📊 KPI Paneli")
                                        
                    k1, k2, k3 = st.columns(3)
                    with k1: st.metric("Fiyat", f"{son_fiyat:.2f}")
                    with k2: st.metric("RSI", f"{son_rsi:.1f}")
                    with k3: st.metric("Potansiyel", f"%{potansiyel:+.2f}")

                    k4, k5, k6 = st.columns(3)
                    with k4: st.metric("Karne", f"{puan}/100")
                    with k5: st.metric("Risk/Getiri (R/G)", rr_text)
                    with k6: st.metric("Güven Skoru", f"%{guven_skoru}")
                                                                                                                                            
                    k7, k8, k9 = st.columns(3)
                    with k7: st.metric("Smart Money", f"{smart_money}/100")
                    with k8: st.metric("Başarı Olasılığı", f"%{basari_olasiligi}")
                    with k9: st.metric("Radar Skoru", f"{genel_not:.1f}/100" if isinstance(genel_not, (int, float)) else f"{genel_not}/100")
                  
                    
                    st.markdown("## 📊 Arz / Talep Analizi")

                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("Alım Hacmi", f"{buy_volume:,.0f}")
                    with c2: st.metric("Satım Hacmi", f"{sell_volume:,.0f}")
                    with c3: st.metric("Flow Index", f"%{flow_index*100:.1f}")

                    st.progress(alim_orani / 100)

                    st.write(f"🟢 Alıcı Baskısı: %{alim_orani:.1f}")
                    st.write(f"🔴 Satıcı Baskısı: %{satim_orani:.1f}")
                    
                    # Gelişmiş Teknik Analiz Bilgi Kutuları
                    st.markdown("## 📈 Gelişmiş Teknik Analiz")
                    g1, g2 = st.columns(2)
                    with g1:
                        st.info(f"Trend Gücü: {trend_gucu}\n\nADX: {adx_yorum}\n\nBollinger: {bb_sinyal}")
                    with g2:
                        st.info(f"Hacim Analizi: {hacim_yorum}\n\nSmart Money: {smart_money}/100\n\nBaşarı Olasılığı: %{basari_olasiligi}")
                        
           
                 
                    # ==============================================================
                    # TAHMİN GRAFİĞİ ÇİZİM BLOĞU (BELLEK SIZINTISI & BOYUT HATALARI GİDERİLDİ)
                    # ==============================================================

                    # =========================
                    # 1. MODEL + TAHMİN GÜVENLİLEŞTİRME
                    # =========================
                    if models is None:
                        st.warning("Model bulunamadı, basit analiz moduna geçiliyor.")
                        proba = 0.5
                        potansiyel = 0
                        tahmin_serisi = np.array([son_fiyat] * 5, dtype=float)
                    else:
                        if tahmin_serisi is None:
                            tahmin_serisi = np.array([son_fiyat] * 5, dtype=float)
                        else:
                            tahmin_serisi = np.array(tahmin_serisi, dtype=float)

                    # NaN temizliği
                    tahmin_serisi = np.nan_to_num(tahmin_serisi, nan=son_fiyat)

                    alt_sinir = np.nan_to_num(
                        np.array(alt_sinir, dtype=float) if alt_sinir is not None else tahmin_serisi * 0.98,
                        nan=son_fiyat * 0.98
                    )

                    ust_sinir = np.nan_to_num(
                        np.array(ust_sinir, dtype=float) if ust_sinir is not None else tahmin_serisi * 1.02,
                        nan=son_fiyat * 1.02
                    )


                    # =========================
                    # 2. VERİ TEMİZLEME (KRİTİK FIX)
                    # =========================
                    kapanis_arr = pd.to_numeric(df["Close"], errors="coerce").dropna()

                    if len(kapanis_arr) == 0:
                        st.error("❌ Fiyat verisi boş veya bozuk!")
                        st.stop()

                    son_veri = min(60, len(kapanis_arr))
                    y_gercek = kapanis_arr.tail(son_veri).to_numpy(dtype=float)

                    if len(y_gercek) == 0:
                        st.error("❌ Grafik verisi oluşturulamadı!")
                        st.stop()

                    x_gercek = np.arange(len(y_gercek))


                    # =========================
                    # 3. GRAFİK OLUŞTURMA
                    # =========================
                    st.markdown("## 📈 Tahmin Grafiği")

                    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#121212")
                    ax.set_facecolor("#1E1E1E")

                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["left"].set_color("white")
                    ax.spines["bottom"].set_color("white")

                    ax.set_title(f"{hisse_kodu} Yapay Zeka Tahmini", color="white", fontsize=14)
                    ax.plot(x_gercek, y_gercek, color="#00F0FF", linewidth=2, label="Gerçek")


                    # =========================
                    # 4. TAHMİN & GÜVEN BANDI BLOĞU (FİNAL FIX)
                    # =========================
                    if len(tahmin_serisi) > 0:
                        # Gerçek verinin son noktasını tahminin başlangıcı yaparak grafikte süreklilik sağlıyoruz
                        tahmin_y = np.concatenate(([y_gercek[-1]], tahmin_serisi))
                        
                        # X eksenini tahmin serisinin uzunluğuna göre dinamik kuruyoruz
                        x_tahmin = np.arange(len(y_gercek) - 1, len(y_gercek) - 1 + len(tahmin_y))

                        ax.plot(
                            x_tahmin,
                            tahmin_y,
                            "--",
                            color="#FF00FF",
                            linewidth=2,
                            label="Tahmin"
                        )

                        # =========================
                        # 5. GÜVEN BANDI ÇİZİMİ
                        # =========================
                        if len(alt_sinir) == len(tahmin_serisi) and len(ust_sinir) == len(tahmin_serisi):
                            base = y_gercek[-1]
                            alt_band = np.concatenate(([base], alt_sinir))
                            ust_band = np.concatenate(([base], ust_sinir))

                            # Olası bir boyut uyuşmazlığına karşı güvenli kırpma
                            min_len = min(len(x_tahmin), len(alt_band), len(ust_band))

                            ax.fill_between(
                                x_tahmin[:min_len],
                                alt_band[:min_len].astype(float),
                                ust_band[:min_len].astype(float),
                                color="#FF00FF",
                                alpha=0.15,
                                label="Güven Aralığı"
                            )
                    else:
                        st.warning("⚠ Tahmin serisi boş!")
                        
                    
                    # =========================
                    # 6. TEKNİK SEVİYELER VE GÖSTERİM
                    # =========================
                    ax.axhline(destek, color="green", linestyle=":", alpha=0.7, label="Destek")
                    ax.axhline(direnc, color="red", linestyle=":", alpha=0.7, label="Direnç")
                    ax.axhline(stop_loss, color="orange", linestyle="--", alpha=0.8, label="Stop Loss")
                    ax.axhline(kar_al, color="lime", linestyle="--", alpha=0.8, label="Kar Al")
                    
                  
                    # Limitleri tahmin serisinin uzunluğuna göre dinamik yapıyoruz (+2 pay bırakarak)
                    ax.set_xlim(0, len(y_gercek) + len(tahmin_serisi) + 2)

                    ax.tick_params(colors="white")
                    ax.grid(True, color="#2D2D2D")
                    ax.set_ylabel("Fiyat (TL)", color="white")
                    ax.set_xlabel("Gün", color="white")
                    ax.legend(loc="upper left")

                    fig.tight_layout()

                    # Streamlit çıktısı ve bellek temizliği
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)

                    # =========================
                    # 8. DEBUG PANEL (CLEAN)
                    # =========================
                    if st.checkbox("Debug bilgileri"):
                        st.write("tahmin_serisi (ilk 5):", tahmin_serisi[:5])
                        st.write("alt_sinir (ilk 5):", alt_sinir[:5])
                        st.write("ust_sinir (ilk 5):", ust_sinir[:5])
                        st.write("kapanis_arr len:", len(kapanis_arr))
                        st.write("y_gercek len:", len(y_gercek))
                        st.write("y_gercek min:", float(np.min(y_gercek)))
                        st.write("y_gercek max:", float(np.max(y_gercek)))
                        st.write("X_live shape:", X_live.shape)
                        st.write("DEBUG tahmin_serisi:", str(tahmin_serisi))
                        st.write("TYPE:", str(type(tahmin_serisi)))
                        st.write("LEN:", len(tahmin_serisi) if tahmin_serisi is not None else "None")


                    # Destek & Direnç Metrikleri Panel Gösterimi
                    st.markdown("## 🎯 Destek / Direnç")
                    d1, d2, d3 = st.columns(3)
                    with d1: st.metric("Destek", f"{destek:.2f}")
                    with d2: st.metric("Fiyat", f"{son_fiyat:.2f}")
                    with d3: st.metric("Direnç", f"{direnc:.2f}")

                    # Yapay Zeka Özet Kartı
                    st.markdown("## 🤖 Yapay Zeka Tahmini")
                    st.info(f"""
                    Hedef Fiyat: {hedef_fiyat:.2f} TL
                    Potansiyel: %{potansiyel:+.2f}
                    Stop Loss: {stop_loss:.2f} TL
                    Kar Al: {kar_al:.2f} TL
                    Risk/Getiri: {rr_text}
                    Tahmin Gücü: {tahmin_gucu}
                    """)

                    # Teknik Özet Listesi
                    st.markdown("## 📋 Teknik Özet")
                    yorumlar = [
                        f"Trend Durumu: {trend_gucu}", 
                        f"Stop Loss: {stop_loss:.2f}", 
                        f"Kar Al: {kar_al:.2f}", 
                        f"Hisse Karnesi: {puan}/100", 
                        f"Risk/Getiri Oranı: {rr_text}"
                    ]
                    for y in yorumlar: 
                        st.write("•", y)

                    # Nihai Karar Renkli Sinyal Kutusu
                    st.markdown("## 🚦 Nihai Karar")
                    st.markdown(f"""
                        <div style="background:{sinyal_rengi}; padding:20px; border-radius:12px; text-align:center; font-size:26px; font-weight:bold; color:white;">
                            {sinyal_metni}
                        </div>
                        """, unsafe_allow_html=True)

                    # Genel Not Gösterimi
                    st.markdown("## 🎖️ Genel Değerlendirme")
                    st.markdown(f"### {derece}  \n📊 Skor: **{genel_not}/100**")

                    # Güven Skoru İlerleme Çubuğu
                    st.markdown("## ⭐ Güven Skoru")
                    st.progress(float(guven_skoru) / 100)
                    st.markdown(f"<h3 style='text-align:center'>%{guven_skoru}</h3>", unsafe_allow_html=True)

                else:
                    st.warning("Hisse verisi boş döndü. Doğru sembol girdiğinizden emin olun.")

            except Exception as e:
                st.error(f"Analiz sırasında beklenmeyen bir hata oluştu: {e}")
                
               
        
               
###################################################################################################################################################

###################################################################################################################################################
         
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
            
          
            import pandas as pd
            import ta

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
                    
                    
                    
                    st.pyplot(fig)
            

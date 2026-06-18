# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 23:35:16 2026

@author: EmirAysu
"""


# =============================================================================
# 1. STANDART KÜTÜPHANELER (Python'ın kendi içindekiler)
# =============================================================================
import os
import sys
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import requests
from types import MappingProxyType

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports import zoneinfo as ZoneInfo  # Eski Python sürümleri için güvence

# Streamlit'in Spyder/Bare modda çalışırken attığı "missing ScriptRunContext" uyarısını sustur
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.CRITICAL)

# =============================================================================
# 2. ÜÇÜNCÜ PARTİ KÜTÜPHANELER (pip ile kurduklarınız)
# =============================================================================
import numpy as np
import pandas as pd
import joblib
import ta
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh




# =============================================================================
# 3. ORTAM TESPİTİ VE GÜVENLİK MUHAFIZLARI (CRITICAL FIX)
# =============================================================================


# Uygulamanın çalıştığı ana klasörü (proje kök dizinini) dinamik olarak bulur
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model yolunu bu ana klasöre göre göreli (relative) olarak inşa eder
MODEL_PATH = os.path.join(BASE_DIR, "models", "ensemble.pkl")


IS_STREAMLIT = False

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is not None:
        IS_STREAMLIT = True
except ImportError:
    pass

# PYQT_AVAILABLE değişkenini projedeki tüm sınıflardan ÖNCE kesin olarak tanımlıyoruz
try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton
    from PyQt5.QtCore import Qt
    PYQT_AVAILABLE = True
except ImportError:
    try:
        from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton
        from PyQt6.QtCore import Qt
        PYQT_AVAILABLE = True
    except ImportError:
        PYQT_AVAILABLE = False

# =============================================================================
# 4. YEREL MODÜLLER (Sizin proje dosyalarınız)
# =============================================================================
try:
    from database import Veritabani
except ImportError:
    class Veritabani:
        def __init__(self): 
            pass


# =============================================================================
# 5. AYARLAR VE SUNUCU UYUMLU DİNAMİK DOSYA YOLLARI (DÜZELTİLMİŞ)
# =============================================================================
FEATURES_OLD = ["Open", "High", "Low", "Close", "Volume"]

FEATURES_FULL = [
    "Open", "High", "Low", "Close", "Volume",
    "ma50", "ma200", "rsi", "macd", "macd_signal", "volatility"
]

MODEL_FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "ma50", "ma200", "rsi", "macd", "macd_signal", "volatility"
]

OUTPUT_SCHEMA = {
    "ai_comment": str,
    "signal": str,
    "score": float
}

# Proje ana dizinini string olarak alıyoruz
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_model_path():
    # 1. Öncelik: Yerel bilgisayardaki sabit D yolu
    if os.path.exists(r"d:\borsa\models\ensemble.pkl"):
        return r"d:\borsa\models\ensemble.pkl"
    
    # 2. Öncelik: Sunucu veya yerel için dinamik küçük harfli yol
    alternative_1 = os.path.join(CURRENT_DIR, "models", "ensemble.pkl")
    if os.path.exists(alternative_1):
        return alternative_1
        
    # 3. Öncelik: Sunucu Linux harf duyarlılığı kontrolü (Büyük 'Models' klasörü ihtimali için)
    alternative_2 = os.path.join(CURRENT_DIR, "Models", "ensemble.pkl")
    if os.path.exists(alternative_2):
        return alternative_2
        
    # 4. Öncelik: Büyük 'Ensemble.pkl' dosya adı ihtimali için
    alternative_3 = os.path.join(CURRENT_DIR, "models", "Ensemble.pkl")
    if os.path.exists(alternative_3):
        return alternative_3

    # Hiçbiri bulunamazsa varsayılan dinamik yolu dön (load_model arayüze hata basacaktır)
    return alternative_1

MODEL_PATH = get_model_path()

# =============================================================================
# VERİ YÖNETİM KATMANI - REFACTOR SAFE
# =============================================================================

def _clean_yfinance_df(df):
    """Yfinance'den gelen verileri temizler ve sütun isimlerini standartlaştırır."""
    if df is not None and not df.empty:
        df = df.copy()
        
        # 1. Aşama: Mükerrer sütun oluştuysa hemen teke düşür
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 2. Aşama: Eğer yfinance sütunları Multi-Index getirdiyse en üst katmanı al
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 3. Aşama: Tüm sütun isimlerini küçük harfe çevir ve boşlukları buda
        df.columns = [str(col).lower().strip() for col in df.columns]
        
        # 4. Aşama: Tarih indeksini sağlama al
        df.index = pd.to_datetime(df.index)
        
    return df

if IS_STREAMLIT:
    @st.cache_data(ttl=60)
    def get_live_data(symbol: str, period="1d", interval="5m"):
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
        return _clean_yfinance_df(df)

    @st.cache_data(ttl=3600)
    def get_history(symbol: str, period="300d", interval="1d"):
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
        return _clean_yfinance_df(df)
else:
    def get_live_data(symbol: str, period="1d", interval="5m"):
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
        return _clean_yfinance_df(df)

    def get_history(symbol: str, period="300d", interval="1d"):
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
        return _clean_yfinance_df(df)

@st.cache_resource(show_spinner=False)
def load_model() -> dict | None:
    # 1. Kontrol: Dosya diskte var mı?
    if not os.path.isfile(MODEL_PATH):
        st.sidebar.error(f"📁 Model dosyası bulunamadı! Aranan yol: {MODEL_PATH}")
        return None
        
    try:
        model = joblib.load(MODEL_PATH)
        
        # 2. Kontrol: Model boş mu döndü?
        if model is None:
            st.sidebar.error("❌ joblib.load boş (None) döndü.")
            return None
            
        # 3. Kontrol: Model dict (sözlük) tipinde mi?
        if not isinstance(model, dict):
            st.sidebar.error(f"⚠️ Yapı Hatası: Model bir 'dict' (sözlük) değil! Yüklenen tip: {type(model)}")
            return None
            
        # 4. Kontrol: İçinde 'xgb' anahtarı var mı?
        if "xgb" not in model:
            st.sidebar.error("⚠️ Yapı Hatası: Sözlük içinde 'xgb' anahtarı bulunamadı.")
            return None
            
        return model
        
    except Exception as e:
        # 5. Kontrol: Kritik bir Python/Kütüphane hatası fırladı mı?
        st.sidebar.error(f"💥 Kritik Yükleme Hatası: {str(e)}")
        return None

# ==============================================================================
# 🛠️ GÜVENLİK VE GECİCİ KORUMA MOTORLARI (EKSİK FONKSİYONLAR İÇİN)
# ==============================================================================

def get_isyatirim_institutional_flow(symbol):
    """
    İş Yatırım takas/kurumsal veri motoru. 
    Fonksiyon sistemde tanımlı değilse ana motorun çökmesini engellemek için None döner.
    """
    try:
        # İleride buraya kurumsal akış kazıma (scraping) kodlarınızı ekleyebilirsiniz.
        return None
    except Exception:
        return None

def get_isyatirim_flow(symbol):
    """İş Yatırım genel para akışı motoru eksikse sistemi çökertmeyen bypass."""
    try:
        return None
    except Exception:
        return None


# ==============================================================================
# 📈 YENİDEN YAPILANDIRILMIŞ VE HATASIZ HİBRİT MOTOR FONKSİYONLARI
# ==============================================================================

def get_isyatirim_price(symbol):
    """İş Yatırım web sitesinden anlık fiyat çeker."""
    try:
        url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/default.aspx"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        
        html_tables = pd.read_html(url, storage_options={"User-Agent": headers})
        if not html_tables:
            return None

        df_prices = html_tables[0]
        df_prices.columns = [str(col).strip() for col in df_prices.columns]

        # Sembolün sonundaki .IS uzantısını temizleyerek arama doğruluğunu sağlıyoruz
        hisse_temiz = symbol.replace(".IS", "").strip().upper()
        symbol_row = df_prices[df_prices.iloc[:, 0].astype(str).str.upper() == hisse_temiz]

        if not symbol_row.empty:
            fiyat_kolonu = "Son (TL)" if "Son (TL)" in df_prices.columns else "Son"
            raw_price = str(symbol_row[fiyat_kolonu].values[0]) if fiyat_kolonu in df_prices.columns else str(symbol_row.iloc[0, 1])
            
            raw_price = raw_price.replace(".", "").replace(",", ".").replace(" ", "")
            return float(raw_price)
    except Exception as e:
        print(f"⚠️ İş Yatırım fiyat çekme hatası ({symbol}): {e}")
        return None
    return None

def get_price_history(symbol, period="1y"):
    """Yahoo Finance üzerinden geçmiş verileri güvenli şekilde indirir."""
    sembol_temiz = symbol.replace(".IS", "").strip().upper()
    return _clean_yfinance_df(yf.download(f"{sembol_temiz}.IS", period=period, auto_adjust=True, progress=False))

def get_live_price_hybrid(symbol):
    """Önce İş Yatırım, başarısız olursa Yahoo."""
    try:
        price = get_isyatirim_price(symbol)
        if price and price > 0:
            return float(price)
    except Exception:
        pass
    return get_live_price(symbol)

def calculate_hybrid_smart_money(symbol, close, flow_signal, last=None):
    """
    Geliştirilmiş, NameError ve Parametre çakışmalarından arındırılmış Akıllı Para Motoru.
    'last' parametresi dışarıdan güvenle kabul edilir, gelmezse varsayılan değer atanır.
    """
    try:
        kurum_verisi = get_isyatirim_institutional_flow(symbol)
        if kurum_verisi is not None:
            # Eğer kurumsal veri bir sözlük (dict) olarak dönüyorsa içinden skoru çeker
            if isinstance(kurum_verisi, dict):
                return float(np.clip(kurum_verisi.get("smart_score", 50), 0, 100))
            return float(np.clip(kurum_verisi, 0, 100))
    except Exception:
        pass
    
    # 🌟 KRİTİK EŞİK: Eğer dışarıdan indikatör sözlüğü (last) gelmediyse, 
    # alt motorun NameError vermemesi için güvenli bir varsayılan veri seti oluşturulur.
    if last is None:
        last = {"rsi": 50, "macd": 0, "macd_signal": 0, "ma200": close}
        
    # Artık 'last' değişkeni Python tarafından kesinlikle tanınıyor!
    return _calculate_smart_money(last=last, close=close, flow_signal=flow_signal)

def get_hybrid_flow(symbol, df):
    """Hibrit Emir Akışı Denetleyicisi."""
    try:
        flow = get_isyatirim_flow(symbol)
        if flow is not None:
            return flow
    except Exception:
        pass
    return _calculate_order_flow(df)

def calculate_data_quality(has_live_price, has_institutional_data):
    score = 50
    if has_live_price:
        score += 25
    if has_institutional_data:
        score += 25
    return score

def determine_trend(adx, rsi, flow_signal):
    if adx >= 30:
        trend_strength = "strong"
    elif adx >= 20:
        trend_strength = "weak"
    else:
        trend_strength = "none"

    score = 0
    if rsi > 55:
        score += 1
    elif rsi < 45:
        score -= 1

    if flow_signal > 0:
        score += 1
    elif flow_signal < 0:
        score -= 1

    if trend_strength == "none":
        return "⚪ Yatay / Belirsiz Rejim"

    if trend_strength == "weak":
        if score >= 1: return "🟡 Zayıf Yükseliş Eğilimi"
        elif score <= -1: return "🟠 Zayıf Düşüş Eğilimi"
        else: return "⚪ Kararsız / Sıkışma"

    if trend_strength == "strong":
        if score >= 2: return "🟢 Güçlü Yükseliş Trendi"
        elif score <= -2: return "🔴 Güçlü Düşüş Trendi"
        elif score == 1: return "🟡 Trend var ama momentum zayıf"
        elif score == -1: return "🟠 Trend aşağı ama alım tepki ihtimali"
        else: return "⚪ Güçlü trend içinde konsolidasyon"

def calculate_exit_strategy(
    price: float,
    rsi: float,
    buy_pressure: float,
    flow_signal: float,
    ma50: float,
    gunluk_getiri: float,
    direnc: float,
    destek: float,
    fake_rally: bool = False
) -> dict:
    """
    Kurumsal para çıkışı, tepe oluşumu ve kâr koruma modülü.
    """
    # Değişkenleri başlangıç değerleri ile tanımla (Güvenli yapı)
    ex_action = "TUT"
    ex_comment = "Belirgin tepe veya dağıtım sinyali bulunmuyor."

    # --- TEPE SKORU HESAPLAMA ---
    tepe_skoru = 0
    if rsi > 70:
        tepe_skoru += 25
    elif rsi > 60:
        tepe_skoru += 10

    if flow_signal < 0:
        tepe_skoru += 25
    if buy_pressure < 0.48:
        tepe_skoru += 25
    if price >= direnc * 0.98:
        tepe_skoru += 25
    if fake_rally:
        tepe_skoru += 20

    tepe_skoru = min(tepe_skoru, 100)

    # --- KAR KORUMA SKORU HESAPLAMA ---
    kar_koruma_skoru = 0
    if rsi > 70:
        kar_koruma_skoru += 30
    if buy_pressure < 0.45:
        kar_koruma_skoru += 20
    if flow_signal < 0:
        kar_koruma_skoru += 40
    if fake_rally:
        kar_koruma_skoru += 15

    kar_koruma_skoru = min(kar_koruma_skoru, 100)

    # --- DURUM ANALİZLERİ ---
    if kar_koruma_skoru < 30:
        kar_koruma_durumu = "🟢 Pozisyonu Koru / Tut"
    elif kar_koruma_skoru < 60:
        kar_koruma_durumu = "🟡 Kârı Koru (Yakın Stop)"
    else:
        kar_koruma_durumu = "🟠 Kademeli Satış Düşünülmeli"

    if tepe_skoru <= 25:
        tepe_bolgesi_durumu = "🟢 Normal Bölge"
    elif tepe_skoru <= 50:
        tepe_bolgesi_durumu = "🟡 Dikkat Seviyesi"
    elif tepe_skoru <= 75:
        tepe_bolgesi_durumu = "🟠 Kâr Koru / İzle"
    else:
        tepe_bolgesi_durumu = "🔴 GÜÇLÜ SATIŞ BÖLGESİ"

    # --- SİNYAL VE ALARMLAR ---
    zirve_yorgunlugu_sinyali = "🟢 Normal (Zirve Baskısı Yok)"
    if price > ma50 * 1.08 and rsi > 65 and flow_signal < 0:
        zirve_yorgunlugu_sinyali = "⚠️ YÜKSELİŞ DEVAM EDİYOR AMA PARA ÇIKIŞI BAŞLADI"

    gun_ici_alarm = "🟢 Normal"
    if gunluk_getiri > 4 and flow_signal < 0:
        gun_ici_alarm = "💰 KÂR KORUMA MODU AKTİF"

    # --- AKSİYON KARARI ---
    if flow_signal < -0.15 and buy_pressure < 0.45:
        ex_action = "BEKLE/SAT"
        ex_comment = "Güçlü satış baskısı mevcut. Para çıkışı devam ediyor."
    elif tepe_skoru >= 75:
        ex_action = "KADEMELİ SAT"
        ex_comment = f"Tepe skoru %{tepe_skoru}. Kâr realizasyonu düşünülebilir."
    elif tepe_skoru >= 50:
        ex_action = "KÂR KORU / İZLE"
        ex_comment = "Yorgunluk belirtileri oluşuyor."

    return {
        "kar_koruma_skoru": round(kar_koruma_skoru),
        "kar_koruma_durumu": kar_koruma_durumu,
        "tepe_skoru": round(tepe_skoru),
        "tepe_bolgesi_durumu": tepe_bolgesi_durumu,
        "zirve_yorgunlugu": zirve_yorgunlugu_sinyali,
        "gun_ici_alarm": gun_ici_alarm,
        "exit_strategy_action": ex_action,
        "exit_strategy_comment": ex_comment,
        "stop_loss": round(destek, 2),
        "take_profit": round(direnc, 2),
    }

def get_live_price(symbol: str) -> float | None:
    """Hisse senedinin güncel fiyatını almaya çalışır."""
    yf_symbol = symbol if symbol.endswith(".IS") else f"{symbol}.IS"
    try:
        ticker = yf.Ticker(yf_symbol)
        
        try:
            if hasattr(ticker, "fast_info"):
                fi = ticker.fast_info
                live_price = fi.get("lastPrice") or fi.get("regularMarketPrice") if isinstance(fi, dict) else getattr(fi, "lastPrice", None) or getattr(fi, "regularMarketPrice", None)
                if live_price is not None and live_price > 0:
                    return float(live_price)
        except Exception:
            pass

        try:
            info = ticker.info
            live_price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
            if live_price is not None and live_price > 0:
                return float(live_price)
        except Exception:
            pass

        df = yf.download(yf_symbol, period="5d", interval="1d", auto_adjust=True, progress=False, threads=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" in df.columns:
                # Seri mi yoksa DataFrame mi döndüğünü garanti altına alıyoruz
                val = df["Close"].iloc[-1]
                close_price = float(val.values[0]) if hasattr(val, "values") else float(val)
                if close_price > 0:
                    return close_price
    except Exception:
        pass
    return None

# ==============================================================
# ANALİZ VE İNDİKATÖR MOTORU
# ==============================================================
def add_indicators(df: pd.DataFrame):
    df = df.copy()
    
    # Sütun isimlerini tamamen temizle ve küçük harfe sabitle
    df = df.loc[:, ~df.columns.duplicated()]
    df.columns = [str(col).lower().strip() for col in df.columns]
    
    # 'close' sütununu güvenli bir şekilde çek
    if "close" in df.columns:
        close_ser = pd.to_numeric(df["close"], errors="coerce").astype(float)
    else:
        close_ser = pd.to_numeric(df.iloc[:, 3], errors="coerce").astype(float)
        
    close_ser = close_ser.ffill().bfill()

    # Hesaplamaları sadece küçük harfli sütunlara yapıyoruz (Pandas çakışmasın diye)
    df["rsi"] = ta.momentum.RSIIndicator(close_ser, window=14, fillna=True).rsi()
    
    macd_obj = ta.trend.MACD(close_ser, fillna=True)
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()

    df["ma50"] = close_ser.rolling(window=50, min_periods=1).mean()
    df["ma200"] = close_ser.rolling(window=200, min_periods=1).mean()

    df["return"] = close_ser.pct_change().fillna(0)
    df["volatility"] = df["return"].rolling(window=10, min_periods=1).std().fillna(0)

    # Tablonun yok olmasını engellemek için boşlukları doldur
    df = df.ffill().bfill()

    # 🌟 SÜTUN KÖPRÜSÜNÜ EN SONDA SÖZLÜKLE YAPIYORUZ (Kritik Hamle)
    # Mevcut küçük harfli sütunların ilk harfi büyük hallerini güvenli bir sözlük yapısıyla ekliyoruz
    new_cols = {}
    for col in df.columns:
        cap_col = str(col).capitalize().strip()
        if cap_col != str(col): # Eğer zaten büyük harfli değilse listeye ekle
            new_cols[cap_col] = df[col]
            
    # Tüm yeni büyük harfli sütunları tek hamlede DataFrame'e dikiyoruz
    df = df.assign(**new_cols)

    return df

def calculate_flow(buy, sell):
    return (buy - sell) / (buy + sell + 1e-9)

def calculate_final_score(proba, guven, smart_money, flow, adx, rsi, hacim_orani):
    score = (
        proba * 0.35 +
        guven * 0.20 +
        smart_money * 0.20 +
        ((flow + 1) / 2) * 0.25
    )
    if adx < 20: score *= 0.85
    if rsi > 80: score *= 0.9
    if hacim_orani < 0.7: score *= 0.95

    return float(np.clip(score, 0, 1))


def detect_regime(adx, rsi, flow_signal):
    """Daha dengeli piyasa rejimi sınıflandırması."""
    # FLOW'u güvenli banda çek (aşırı tepkiyi engeller)
    flow = max(min(flow_signal, 0.2), -0.2)

    # ---------------------------------------------------------
    # 1. GÜÇLÜ TRENDLER (Öncelikli Dar Kapsamlı Koşullar)
    # ---------------------------------------------------------
    if adx > 30 and rsi < 40 and flow < -0.06:
        return "🔴 Güçlü Düşüş Trendi"

    if adx > 25 and rsi > 55 and flow > 0.06:
        return "🟢 Güçlü Yükseliş Trendi"

    # ---------------------------------------------------------
    # 2. ZAYIF TRENDLER VE YATAY PİYASA (Geniş Kapsamlı Koşullar)
    # ---------------------------------------------------------
    if adx > 25 and flow < -0.03:
        return "🔴 Zayıf Düşüş Eğilimi"

    if adx < 20 and -0.03 <= flow <= 0.03 and 40 <= rsi <= 55:
        return "⚪ Yatay / Denge"

    if flow > 0.03 and rsi > 45:
        return "🟢 Zayıf Yükseliş"

    # ---------------------------------------------------------
    # 3. FALLBACK (Hiçbir Koşula Uymayan Durum)
    # ---------------------------------------------------------
    return "⚪ Kararsız"


def get_signal(adx, rsi, flow_signal, smart_money=50):
    # Nötr başlangıç skoru
    score = 50

    # 1. RSI Momentum ve Trend Yönü Belirleme
    if rsi > 55:
        score += 10
        score += 10 if adx > 25 else -5
    elif rsi < 40:
        score -= 10
        score -= 10 if adx > 25 else -5
    else:
        if adx > 25:
            score -= 5

    # 2. Flow Etkisi (Küçük ama stabil)
    score += flow_signal * 100 * 0.6

    # 3. Smart Money Etkisi
    score += (smart_money - 50) * 0.4

    # 4. Clamp (Skoru 0-100 arasında sınırla)
    score = max(0, min(100, score))

    # ---------------------------------------------------------
    # FINAL SIGNAL
    # ---------------------------------------------------------
    if score >= 70:
        return "🟢 AL", score
    if score <= 40:
        return "🔴 SAT", score
    
    return "⚪ BEKLE", score


def _prepare_data(df: pd.DataFrame, features: list | None, model: dict | None) -> tuple[pd.DataFrame, list]:
    """Analiz öncesi veri hazırlama sürecini yönetir."""
    import streamlit as st
    import numpy as np
    import pandas as pd

    if df is None or df.empty:
        st.error("🚨 _prepare_data Giriş Hatası: Gelen df boş veya None!")
        return pd.DataFrame(), []

    df = df.copy()
    
    # -------------------------------------------------------------------------
    # Adım A: Sütun Temizliği ve Normalizasyon
    # -------------------------------------------------------------------------
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        cleaned_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                cleaned_cols.append(str(col[0]).lower().strip())
            else:
                cleaned_cols.append(str(col).lower().strip())
        df.columns = cleaned_cols
    except Exception as e:
        st.error(f"🚨 _prepare_data Adım A (Sütun Temizliği) Hatası: {e}")
        return pd.DataFrame(), []

    # -------------------------------------------------------------------------
    # Adım B: Numerik Tipe Zorlama
    # -------------------------------------------------------------------------
    try:
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                st.warning(f"⚠️ _prepare_data: Temel sütun eksik: {col}")
    except Exception as e:
        st.error(f"🚨 _prepare_data Adım B (Numerik Zorlama) Hatası: {e}")
        return pd.DataFrame(), []

    # -------------------------------------------------------------------------
    # Adım C: İndikatör Ekleme Kontrolü
    # -------------------------------------------------------------------------
    try:
        df = add_indicators(df)
        if df is None or df.empty:
            st.error("🚨 _prepare_data Adım C: add_indicators fonksiyonu boş tablo döndürdü!")
            return pd.DataFrame(), []
    except Exception as e:
        st.error(f"🚨 _prepare_data Adım C (add_indicators çağrısı) Çöktü: {e}")
        return pd.DataFrame(), []

    # -------------------------------------------------------------------------
    # Adım D: Boşluk Temizleme (dropna) Güvenlik Duvarı
    # -------------------------------------------------------------------------
    try:
        df = df.ffill().bfill()
        df_test = df.replace([np.inf, -np.inf], np.nan).dropna()
        if not df_test.empty:
            df = df_test
        else:
            st.warning("⚠️ _prepare_data: .dropna() sonrası tablo boşalacaktı, ffill koruması devreye girdi.")
    except Exception as e:
        st.error(f"🚨 _prepare_data Adım D (dropna) Hatası: {e}")
        return pd.DataFrame(), []

    # -------------------------------------------------------------------------
    # Adım E: Model Özellikleri (Features) Hizalama
    # -------------------------------------------------------------------------
    try:
        features = [] if features is None else list(features)

        if model is not None and isinstance(model, dict) and "xgb" in model:
            xgb_model = model["xgb"]
            if hasattr(xgb_model, "feature_names") and xgb_model.feature_names:
                features = list(xgb_model.feature_names)
            elif hasattr(xgb_model, "n_features_in_"):
                expected_n = xgb_model.n_features_in_
                if len(features) > expected_n:
                    features = features[:expected_n]

        features = [str(f).lower().strip() for f in features]

        for feature in features:
            if feature not in df.columns:
                df[feature] = 0.0

        valid_features = [f for f in features if f in df.columns]
        return df, valid_features

    except Exception as e:
        st.error(f"🚨 _prepare_data Adım E (Feature Yönetimi) Hatası: {e}")
        return pd.DataFrame(), []


def _calculate_order_flow(df):
    """Hisse senedi veri setindeki hacim ve fiyat hareketlerini inceleyerek
    kurumsal para akışını (Order Flow) hesaplar.
    """
    import numpy as np
    import pandas as pd

    df = df.copy()
    
    # 🌟 1. ADIM: Tabloda sinsice birikmiş mükerrer (duplicate) sütunları tek hamlede ezilerek teke düşürülüyor
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Sütun isimlerini küçük harfe normalize et
    df.columns = [str(col).lower().strip() for col in df.columns]

    # 🌟 2. ADIM: Sütunların birden fazla gelme ihtimaline karşı .iloc ve squeeze() zırhı çekiliyor
    if "close" in df.columns:
        # squeeze() eğer hala çoklu sütun kalmışsa onu tek boyuta zorlar, iloc ise ilk yakaladığı close'u alır
        close_series = df["close"]
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
        close_vals = pd.to_numeric(close_series, errors="coerce").ffill().bfill().values
    else:
        close_vals = pd.to_numeric(df.iloc[:, 3], errors="coerce").ffill().bfill().values

    if "volume" in df.columns:
        volume_series = df["volume"]
        if isinstance(volume_series, pd.DataFrame):
            volume_series = volume_series.iloc[:, 0]
        volume_vals = pd.to_numeric(volume_series, errors="coerce").ffill().bfill().values.astype(float)
    else:
        volume_vals = pd.to_numeric(df.iloc[:, 4], errors="coerce").ffill().bfill().values.astype(float)

    # -------------------------------------------------------------------------
    # GERİ KALAN MATEMATİKSEL HESAPLAMA KODLARI AYNEN DEVAM EDİYOR...
    # -------------------------------------------------------------------------
    price_diff = np.diff(close_vals)
    price_diff = np.insert(price_diff, 0, 0.0)

    buy_mask = price_diff > 0
    sell_mask = price_diff < 0

    buy_volume_total = float(np.sum(volume_vals[buy_mask]))
    sell_volume_total = float(np.sum(volume_vals[sell_mask]))
    total_volume_cum = buy_volume_total + sell_volume_total + 1e-9

    buy_pressure = buy_volume_total / total_volume_cum
    sell_pressure = sell_volume_total / total_volume_cum

    flow_raw = (buy_volume_total - sell_volume_total) / total_volume_cum
    flow_index = float(np.tanh(flow_raw))

    signed_volume = np.where(price_diff > 0, volume_vals, np.where(price_diff < 0, -volume_vals, 0))
    signed_series = pd.Series(signed_volume, dtype="float64").reset_index(drop=True)

    flow_smooth = signed_series.rolling(3, min_periods=1).mean().iloc[-1]
    flow_smooth = 0.0 if pd.isna(flow_smooth) else float(flow_smooth)

    rolling_std = signed_series.rolling(3, min_periods=1).std().iloc[-1]
    rolling_std = 1e-4 if (pd.isna(rolling_std) or rolling_std < 1e-6) else float(rolling_std)

    flow_smooth_norm = np.tanh(flow_smooth / rolling_std)
    flow_signal = float(0.6 * flow_index + 0.4 * flow_smooth_norm)

    todays_total_volume = float(volume_vals[-1]) if len(volume_vals) > 0 else 0.0

    buy_volume = float(todays_total_volume * buy_pressure)
    sell_volume = float(todays_total_volume * sell_pressure)

    vol_series = pd.Series(volume_vals, dtype="float64").reset_index(drop=True)
    avg_volume_20 = float(vol_series.rolling(20, min_periods=1).mean().iloc[-1])
    
    last_volume = todays_total_volume
    last_price_change = float(price_diff[-1]) if len(price_diff) > 0 else 0.0
    
    kalicilik_durumu = "🟢 NORMAL"
    suni_kod = 0

    if last_price_change > 0 and last_volume < avg_volume_20 * 0.70 and buy_pressure < 0.52:
        kalicilik_durumu = "⚠️ SUNİ YÜKSELİŞ"
        suni_kod = 1
    elif last_price_change < 0 and last_volume < avg_volume_20 * 0.70 and sell_pressure < 0.52:
        kalicilik_durumu = "⚠️ SUNİ DÜŞÜŞ"
        suni_kod = -1
    elif last_price_change > 0 and last_volume > avg_volume_20 * 1.30 and buy_pressure > 0.55:
        kalicilik_durumu = "🚀 HACİM ONAYLI YÜKSELİŞ"
        suni_kod = 2

    if flow_signal <= -0.6:
        flow_state = "🔴 GÜÇLÜ SATIŞ"
    elif flow_signal <= -0.2:
        flow_state = "🟠 SATIŞ BASKISI"
    elif flow_signal < 0.2:
        flow_state = "⚪ DENGELİ"
    elif flow_signal < 0.6:
        flow_state = "🟢 ALIM BASKISI"
    else:
        flow_state = "🚀 AGRESİF ALIM"

    return {
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "buy_pressure": buy_pressure,
        "sell_pressure": sell_pressure,
        "flow_raw": flow_raw,
        "flow_index": flow_index,
        "flow_signal": flow_signal,
        "flow_smooth": flow_smooth_norm,
        "flow_state": flow_state,
        "kalicilik_durumu": kalicilik_durumu,
        "suni_kod": suni_kod,
    }


def _calculate_smart_money(last, close, flow_signal):
    """Akıllı para endeksini trend gücü (ADX) ve RSI ile dinamik olarak hesaplar."""
    import pandas as pd
    import numpy as np

    # Seri kilitlenmelerini kıran iç fonksiyon
    def _to_pure_float(val, default=0.0):
        if val is None:
            return float(default)
        if hasattr(val, "values"):
            if len(val.values) > 0:
                flat_vals = np.asarray(val.values).ravel()
                return float(flat_vals[0]) if len(flat_vals) > 0 else float(default)
            return float(default)
        try:
            return float(val)
        except (ValueError, TypeError):
            return float(default)

    # 1. Veri Ayıklama Seansı
    rsi_raw = None
    adx_raw = None

    if hasattr(last, "get"):
        rsi_raw = last.get("rsi") if last.get("rsi") is not None else last.get("RSI")
        adx_raw = last.get("adx") if last.get("adx") is not None else last.get("ADX")
    elif isinstance(last, (pd.Series, pd.DataFrame)):
        last_clean = last.copy()
        if hasattr(last_clean, "index") and hasattr(last_clean.index, "str"):
            last_clean.index = last_clean.index.str.lower().str.strip()
        if "rsi" in last_clean.index:
            rsi_raw = last_clean["rsi"]
        if "adx" in last_clean.index:
            adx_raw = last_clean["adx"]

    # Saf float dönüşümleri
    rsi = _to_pure_float(rsi_raw, default=50.0)
    adx = _to_pure_float(adx_raw, default=25.0)
    flow_val = _to_pure_float(flow_signal, default=0.0)

    # 🌟 2. ADX ENTEGRELİ DİNAMİK SMART MONEY MATEMATİĞİ
    # Temel Skor: Para akış gücüne dayanır (0 ile 100 arasına yayılır)
    base_flow_score = 50.0 + (flow_val * 40.0)

    # RSI Dengesi: Aşırı alım/satım bölgelerine göre kurumsal baskı çarpanı
    rsi_bias = (rsi - 50.0) * 0.3

    # Trend Çarpanı (ADX Kaldıracı): Trend gücü ne kadar yüksekse, 
    # kurumsal paranın bıraktığı iz (skor) o kadar belirginleşir.
    # ADX > 25 ise çarpanı büyütür, zayıf trendlerde skoru nötralize (50'ye doğru) eder.
    if adx > 25:
        trend_multiplier = 1.2
    elif adx < 20:
        trend_multiplier = 0.7
    else:
        trend_multiplier = 1.0

    # Nihai Skoru Harmanla
    smart_money_score = 50.0 + ((base_flow_score + rsi_bias - 50.0) * trend_multiplier)
    
    # Skor Sınırlandırması (0-100 dışına taşmasın)
    smart_money_score = max(0.0, min(100.0, smart_money_score))
    
    return float(smart_money_score)

def _determine_regime(adx, rsi, flow_signal):
    """Piyasa yapısını anahtar kelimelere göre sınıflandırır."""
    if adx > 25 and flow_signal > 0: regime = "TREND_UP"
    elif adx > 25 and flow_signal < 0: regime = "TREND_DOWN"
    elif adx < 18: regime = "SIDEWAYS"
    elif rsi > 70 and flow_signal < 0: regime = "DISTRIBUTION"
    elif rsi < 45 and flow_signal > 0: regime = "ACCUMULATION"
    else: regime = "UNKNOWN"
        
    regime_map = {
         "TREND_UP": "🟢 Güçlü Yükseliş Trendi",
         "TREND_DOWN": "🔴 Güçlü Düşüş Trendi",
         "SIDEWAYS": "⚪ Yatay Piyasa",
         "ACCUMULATION": "🟢 Toplanma Bölgesi",
         "DISTRIBUTION": "🔴 Dağıtım Bölgesi",
         "VOLATILE": "⚡ Yüksek Volatilite",
         "UNKNOWN": "⚪ Belirsiz"
    }
    return regime_map.get(regime, regime)


def is_market_open(strict: bool = True) -> bool:
    """BIST işlem saatlerine göre market açık mı kontrolü."""
    now = datetime.now(ZoneInfo("Europe/Istanbul"))

    if now.weekday() >= 5:  # Hafta sonu
        return False

    current_minutes = now.hour * 60 + now.minute

    pre_open_start = 9 * 60 + 40  # 09:40
    market_open = 10 * 60         # 10:00
    market_close = 18 * 60 + 10   # 18:10

    if strict:
        return market_open <= current_minutes <= market_close

    return pre_open_start <= current_minutes <= market_close


def run_mega_radar_analysis(symbol: str, model: dict, features: list) -> dict | None:
    """
    Ana analiz motoru:
    - Veri indirir
    - Temizler (MultiIndex korumalı)
    - Intraday volume estimation üretir (Leakage-free & Güvenli)
    - Feature engineering pipeline çalıştırır
    - XGBoost ile yön/fiyat tahmini yapar
    """
    market_open = is_market_open(strict=True)

    # =========================================================
    # SYMBOL CLEANING (Giriş Temizliği)
    # =========================================================
    hisse_temiz = symbol.strip().upper().replace(".IS", "")
    sembol = f"{hisse_temiz}.IS"

    # =========================================================
    # 1. DATA DOWNLOAD (Veri İndirme)
    # =========================================================
    df_raw = yf.download(
        sembol,
        period="60d",
        interval="1d",
        progress=False,
        auto_adjust=True
    )

    if df_raw is None or df_raw.empty:
        df_raw = yf.download(hisse_temiz, period="60d", interval="1d", progress=False, auto_adjust=True)
        if df_raw is None or df_raw.empty:
            return None

    # =========================================================
    # 2. MULTIINDEX FIX + CLEAN (Çoklu İndeks Temizliği)
    # =========================================================
    if hasattr(df_raw.columns, "nlevels") and df_raw.columns.nlevels > 1:
        df_raw.columns = df_raw.columns.get_level_values(0)

    # Dış bağımlılık fonksiyon güvenliği
    try:
        df = _clean_yfinance_df(df_raw).copy()
    except NameError:
        df = df_raw.copy()

    # =========================================================
    # 3. INTRADAY VOLUME ESTIMATION & TIME FEATURE (FIXED)
    # =========================================================
    df["estimated_daily_volume"] = df["Volume"].astype(float)
    df["progress"] = 1.0 
    progress = 1.0 

    if market_open:
        now = datetime.now(ZoneInfo("Europe/Istanbul"))
        market_open_time = now.replace(hour=9, minute=40, second=0, microsecond=0)

        minutes_elapsed = max(0, (now - market_open_time).seconds / 60)
        progress = min(minutes_elapsed / 390, 1.0)

        df.loc[df.index[-1], "progress"] = progress

        # Daha stabil non-linear gün içi hacim eğrisi (Sinüs düzeltmesi)
        expected_share = (0.30 * np.sin(np.pi * progress) + 0.70 * progress)
        expected_share = max(expected_share, 0.05)

        last_idx = df.index[-1]
        last_volume = float(df["Volume"].iloc[-1])
        df.loc[last_idx, "estimated_daily_volume"] = last_volume / expected_share

    # =========================================================
    # 4. FEATURE PIPELINE (İndikatörler & Özellikler)
    # =========================================================
    df_prepared, valid_features = _prepare_data(df, features, model)

    if df_prepared.empty or not valid_features:
        return None

    # =========================================================
    # 5. MODEL CHECK (XGBoost Kontrolü)
    # =========================================================
    if "xgb" not in model or not hasattr(model["xgb"], "predict"):
        return None

    # =========================================================
    # 6. SAFE LAST ROW (Güvenli Son Satır Hizalaması)
    # =========================================================
    try:
        last_row = (
            df_prepared
            .reindex(columns=valid_features)
            .iloc[[-1]]
            .fillna(0)
        )
    except Exception:
        return None

    # =========================================================
    # 7. PREDICTION (Yapay Zeka Tahmini)
    # =========================================================
    prediction = model["xgb"].predict(last_row)
    pred_value = float(prediction[0]) if hasattr(prediction, "__len__") else float(prediction)

    # =========================================================
    # 8. OUTPUT (Çıktı Blok Yapısı)
    # =========================================================
    return {
        "symbol": hisse_temiz,
        "prediction": pred_value,
        "market_open": market_open,
        "progress": progress,
        "estimated_daily_volume": float(df["estimated_daily_volume"].iloc[-1])
    }


def no_trade_response():
    return {"signal": "NO_TRADE", "note": "Market closed"}


def _get_market_regime_and_trend(adx, rsi, flow_signal, regime_text):
    """ADX ve RSI değerlerine göre piyasa rejimini ve trend metnini belirler."""
    if adx < 20:
        regime = "range"
    elif adx > 25:
        regime = "trend"
    else:
        regime = "weak_trend"

    if adx > 25 and rsi > 50:
        trend_text = "🟢 Yükseliş Eğilimi"
    elif adx > 25 and rsi < 50:
        trend_text = "🔴 Düşüş Eğilimi"
    else:
        trend_text = "⚪ Yatay / Kararsız"
        
    return regime, trend_text


def _calculate_risk_metrics(df, close, veri_donuk):
    """ATR ve fiyat yapısını hibrit kullanarak destek, direnç ve R/R oranını hesaplar."""
    try:
        import ta
        # Nesne yönelimli ATR çağrısı hatasızlaştırıldı
        atr_indicator = ta.volatility.AverageTrueRange(
            high=df["High"], low=df["Low"], close=df["Close"], window=14
        )
        atr = atr_indicator.average_true_range().iloc[-1]
    except Exception:
        atr = np.nan

    risk = 0.01
    reward = 0.01
    piyasa_notu = ""

    if veri_donuk:
        destek = close
        direnc = close
        rr = 1.0
        piyasa_notu = "Piyasa veri donuk"
    else:
        destek = float(df["Low"].tail(20).min())
        direnc = float(df["High"].tail(20).max())

        price_risk = abs(close - destek)
        atr = np.nan_to_num(atr, nan=price_risk)

        risk = max(atr, price_risk)
        reward = abs(direnc - close)

        risk = max(risk, 0.001)
        rr = reward / risk if risk > 0 else 1.0

    rr = np.clip(rr, 0.0, 5.0)
    return destek, direnc, rr, risk, reward, piyasa_notu


def _get_edge_case_comment(rsi, flow_index, regime):
    """RSI ve Para Akışı uç durumlarına göre kritik uyarı mesajı üretir."""
    if "Düşüş" in regime:
        edge_comment = "🔴 Satış baskısının hakim olduğu piyasa koşulu. "
    elif "Yükseliş" in regime:
        edge_comment = "🚀 Alıcıların hakim olduğu piyasa koşulu. "
    else:
        edge_comment = "🟢 Normal piyasa koşulu. "

    if rsi < 25 and flow_index < 0:
        edge_comment += "⚠️ Aşırı satım + para çıkışı devam ediyor. Panik satış / trend devam riski var."
    elif rsi < 25 and flow_index > 0:
        edge_comment += "🟢 Aşırı satım + para girişi var. Rebound ihtimali artıyor."
    elif rsi > 70 and flow_index < 0:
        edge_comment += "⚠️ Aşırı alım + dağıtım riski oluşuyor."
    return edge_comment


def _calculate_scores_and_signal(proba, smart_money, rsi, flow_index, regime, regime_text, buy_pressure):
    """Ağırlık matrisini çalıştırarak skorları üretir ve sinyali belirler."""
    rsi_weight = 0.20
    flow_weight = 0.20

    if regime == "range":
        rsi_weight = 0.35
        flow_weight = 0.15
    elif regime == "trend":
        rsi_weight = 0.15
        flow_weight = 0.35

    flow_score = (flow_index + 1) / 2

    score = (
        proba * 0.35 +
        (smart_money / 100) * 0.25 +
        (rsi / 100) * rsi_weight +
        flow_score * flow_weight
    )

    if rsi < 25 and flow_index < 0:
        score += 0.05

    if regime == "range":
        score += 0.02 if rsi < 30 else 0
    elif regime == "trend":
        score += 0.02 if flow_index > 0 else 0

    score = np.clip(score, 0, 1)

    if score >= 0.80:
        signal, color = "🚀 ÇOK GÜÇLÜ AL", "#00C853"
    elif score >= 0.70:
        signal, color = "🟢 AL", "#4CAF50"
    elif score >= 0.60:
        signal, color = "🟡 İZLE", "#FFD600"
    elif score >= 0.50:
        signal, color = "⚪ NÖTR", "#90A4AE"
    else:
        signal, color = "🔴 SAT", "#E53935"

    if "Yükseliş" in regime_text and buy_pressure > 0.54:
        if signal == "🔴 SAT":
            signal, color = "🟡 İZLE (POTANSİYEL AL)", "#FFD600"

    return score, signal, color


def _generate_ai_comment(signal, edge_comment, trend_text, flow,
                         dynamic_stop, dynamic_tp, destek, direnc,
                         adx=None, rsi=None, flow_signal=None):
    
    if any(buy_sig in signal for buy_sig in ["🚀 ÇOK GÜÇLÜ AL", "🟢 AL"]):
        return (
            f"{edge_comment}"
            f"Hisse {trend_text} içerisinde güçlü görünüm sergiliyor. "
            f"Alıcı baskısı %{flow['buy_pressure'] * 100:.1f}. "
            f"Stop: {dynamic_stop:.2f} | Hedef: {dynamic_tp:.2f}"
        )

    elif "POTANSİYEL AL" in signal:
        return (
            f"{edge_comment}"
            f"Trend tamamen bozulmuş değil. "
            f"Alıcı baskısı %{flow['buy_pressure'] * 100:.1f}. "
            f"Direnç bölgesi dikkatle takip edilmeli."
        )

    elif "İZLE" in signal:
        return (
            f"{edge_comment}"
            f"Piyasada kararsızlık mevcut. "
            f"Destek {destek:.2f} / Direnç {direnc:.2f} takip edilmeli."
        )

    elif "NÖTR" in signal:
        if "Düşüş" in trend_text:
            return (
                f"{edge_comment}"
                f"Trend aşağı yönlü ancak satış baskısı henüz güçlü SAT sinyali oluşturacak seviyede değil. "
                f"Yeni alım için erken, mevcut pozisyonlar destek bölgesi izlenerek yönetilmeli."
            )
        elif "Yükseliş" in trend_text:
            return (
                f"{edge_comment}"
                f"Trend yukarı yönlü ancak momentum zayıflamış durumda. Direnç kırılımı beklenmeli."
            )
        return (
            f"{edge_comment}"
            f"Piyasa yön arayışında. Net kırılım beklenmeli."
        )
    else:
        return (
            f"{edge_comment}"
            f"Satış baskısı güçlü (%{flow['sell_pressure'] * 100:.1f}). "
            f"Risk yönetimi ön planda tutulmalı."
        )


def institutional_signal_engine(
    df: pd.DataFrame,
    adx: float,
    rsi: float,
    flow_signal: float,
    buy_pressure: float,
    sell_pressure: float,
    close: float,
    flow_index: float = 0.0,
    smart_money: float = 50.0,
    fake_rally: bool = False,
) -> dict:
    """Kurumsal Para Akışı, Trend Rejimi ve Sinyal Üretim Motoru.

    Gelişmiş risk cezaları, dinamik boğa/ayı/nötr trend rejim yönetimi,
    normalize edilmiş güven endeksi, dengelenmiş trend gücü, trend eğilimi
    koruma filtreleri ve ayı piyasası kesin bloklama kalkanı içerir.
    """

    # =====================================================
    # 1. VOLATİLİTE HESAPLAMA
    # =====================================================
    volatility = df["Close"].pct_change().rolling(10).std().iloc[-1]
    if np.isnan(volatility):
        volatility = 0.0

    # =====================================================
    # 2. DİNAMİK REJİM BELİRLEME (GRI ALAN FİLTRELİ)
    # =====================================================
    if adx >= 30:
        if flow_index > 0.10:
            regime = "BOĞA_TREND"  # Net para girişiyle desteklenen yükseliş trendi
        elif flow_index < -0.10:
            regime = "AYI_TREND"   # Net para çıkışıyla kesinleşen düşüş trendi
        else:
            regime = "TREND"        # Güçlü ama para akışı yönünden nötr/kararsız trend
    elif adx <= 20:
        regime = "YATAY"           # Menzil içi konsolidasyon (Range)
    else:
        regime = "GEÇİŞ"           # Kararsız / Kurulum bölgesi

    # =====================================================
    # 3. HAM ALT SKORLAR (0 - 100 NORMALİZASYONU)
    # =====================================================
    trend_score = np.clip(adx * 2.0, 0, 100)
    momentum_score = np.clip(100 - abs(rsi - 55) * 2.5, 0, 100)

    # flow_signal (-1 ile +1) aralığını 0-100 arasına lineer çeker
    flow_signal = np.clip(flow_signal, -1.0, 1.0)
    flow_score = np.clip(
        (((flow_signal + 1) / 2) * 70 + buy_pressure * 30), 0, 100
    )
    smart_money_score = np.clip(smart_money, 0, 100)

    # =====================================================
    # 4. REJİME GÖRE DİNAMİK AĞIRLIKLANDIRMA
    # =====================================================
    if regime == "BOĞA_TREND" or regime == "TREND":
        w_trend, w_flow, w_smart, w_momentum = 0.35, 0.30, 0.20, 0.15
        
    elif regime == "AYI_TREND":
        w_trend, w_flow, w_smart, w_momentum = 0.20, 0.40, 0.30, 0.10
        
    elif regime == "YATAY":
        w_trend, w_flow, w_smart, w_momentum = 0.15, 0.30, 0.25, 0.30
        
    else:  # GEÇİŞ
        w_trend, w_flow, w_smart, w_momentum = 0.25, 0.30, 0.25, 0.20

    # Ağırlıklı ham skor hesaplama
    score = (
        trend_score * w_trend
        + flow_score * w_flow
        + smart_money_score * w_smart
        + momentum_score * w_momentum
    )

    # =====================================================
    # 5. RİSK CEZALARI (MARKET RISK PENALTIES)
    # =====================================================
    spread_risk = abs(buy_pressure - sell_pressure)
    risk_penalty = 0

    if volatility > 0.05:
        risk_penalty += 12
    elif volatility > 0.03:
        risk_penalty += 6

    if spread_risk > 0.35:
        risk_penalty += 10
    elif spread_risk > 0.20:
        risk_penalty += 5

    score -= risk_penalty

    # =====================================================
    # 6. SİSTEMİK BONUSLAR (INSTITUTIONAL SUPPORT)
    # =====================================================
    bonus = 0
    if smart_money > 80:
        bonus += 12
    elif smart_money > 70:
        bonus += 8
    elif smart_money > 60:
        bonus += 4

    if flow_index > 0.30:
        bonus += 8
    elif flow_index > 0.15:
        bonus += 4

    score += bonus

    # =====================================================
    # 7. YAPISAL CEZALAR (MAL DAĞITIM VE TUZAK FİLTRELERİ)
    # =====================================================
    penalty = 0

    if smart_money < 40:
        penalty += 12
    elif smart_money < 50:
        penalty += 6

    if flow_index < -0.30:
        penalty += 10
    elif flow_index < -0.15:
        penalty += 5

    # --- Kurumsal Kaçış Kombinasyonları ---
    if smart_money < 40 and flow_index < -0.15:
        penalty += 15

    if flow_index < -0.20 and buy_pressure < 0.45:
        penalty += 10

    # Yalancı yükseliş cezaları
    if fake_rally:
        penalty += 10

    if fake_rally and flow_index < 0:
        penalty += 10

    score -= penalty

    # --- Trend Eğilimi Cezası (Trend Bias Penalty) ---
    if regime == "AYI_TREND" and score > 75:
        score -= 10

    score = float(np.clip(score, 0, 100))

    # =====================================================
    # 8. CONFIDENCE (GÜVEN) HESAPLAMA
    # =====================================================
    adx_score = min(100.0, adx * 2.0)
    
    flow_conf = ((flow_signal + 1) / 2) * 100
    flow_conf = np.clip(flow_conf, 0, 100)
    
    vol_conf = np.clip(100 - volatility * 1000, 0, 100)

    confidence = (
        adx_score * 0.30 +
        flow_conf * 0.30 +
        smart_money * 0.25 +
        vol_conf * 0.15
    )

    if fake_rally:
        confidence -= 10

    confidence = float(np.clip(confidence, 0, 100))

    # =====================================================
    # 9. TREND GÜCÜ VE METİNSEL ANLAMLADIRMA
    # =====================================================
    flow_component = max(0.0, flow_index) * 100.0

    trend_power = (
        adx_score * 0.45 +
        flow_component * 0.30 +
        smart_money * 0.25
    )
    if fake_rally:
        trend_power -= 10
    trend_power = float(np.clip(trend_power, 0, 100))
    if trend_power >= 80:
        trend_text = "🟢 Çok Güçlü Trend"
    elif trend_power >= 60:
        trend_text = "🟡 Güçlü Trend"
    elif trend_power >= 35:
        trend_text = "⚪ Kararsız"
    else:
        trend_text = "🔴 Zayıf Trend"

    # =====================================================
    # 10. UYUMSUZLUK (DIVERGENCE) TESPİTİ
    # =====================================================
    divergence = "YOK"
    if rsi < 40 and flow_signal > 0:
        divergence = "🟢 Yükseliş Uyumsuzluğu"
    elif rsi > 65 and flow_signal < 0:
        divergence = "🔴 Düşüş Uyumsuzluğu"

    # =====================================================
    # 11. MATRİS SİNYAL ÜRETİMİ (ÇİFT KATMANLI GÜVEN FİLTRELİ)
    # =====================================================
    if confidence < 20:
        signal = "⚪ BEKLE"
    elif confidence < 35 and score >= 75:
        signal = "⚪ BEKLE"
    elif score >= 85:
        signal = "🚀 MEGA AL"
    elif score >= 75:
        signal = "🟢 GÜÇLÜ AL"
    elif score >= 60:
        signal = "🟡 AL"
    elif score >= 45:
        signal = "⚪ BEKLE"
    elif score >= 30:
        signal = "🟠 SAT"
    else:
        signal = "🔴 GÜÇLÜ SAT"

    # --- AYI PİYASASI HARD BLOCK KALKANI (YENİ EKLEME) ---
    # Rejim net bir düşüş trendiyse, agresif alım sinyalleri kesin olarak engellenir
    if regime == "AYI_TREND" and signal in ["🚀 MEGA AL", "🟢 GÜÇLÜ AL"]:
        signal = "⚪ BEKLE"

    return {
        "sinyal": signal,
        "skor": round(score, 2),
        "güven": round(confidence, 2),
        "trend_gucu": round(trend_power, 2),
        "rejim": regime,
        "trend_text": trend_text,
        "trend_skoru": round(trend_score, 2),
        "momentum_skoru": round(momentum_score, 2),
        "akış_skoru": round(flow_score, 2),
        "smart_money": round(smart_money, 2),
        "uyumsuzluk": divergence,
        "volatilite": round(volatility * 100, 2),
    }

def get_grade(score):
    try:
        # 🟢 KRİTİK DÜZELTME: Gelen veri string ("52.25") bile olsa sayıya çeviriyoruz
        score = float(score)
    except (ValueError, TypeError):
        # Eğer veri bozuk veya None gelirse sistem çökmesin, en düşük notu ver
        return "F"
    # 📊 BORSA İSTANBUL GERÇEKLERİNE GÖRE YENİDEN AYARLANMIŞ FINANSAL BAREM
    if score >= 80: 
        return "A+"  # Olağanüstü Güçlü Bölge
    elif score >= 70: 
        return "A"   # Güçlü Boğa Trendi
    elif score >= 58: 
        return "B"   # Pozitif / Yükseliş Eğilimli
    elif score >= 48: 
        return "C"   # GEÇİŞ / DENGE BÖLGESİ (52.25 artık tam buraya oturacak!)
    elif score >= 35: 
        return "D"   # Zayıf / Ayı Baskısı Altında
    else: 
        return "F"   # Ağır Kusurlu / Çöküş Trendi   


def final_decision(signal, exit_signal):
    """Sinyal içeriğindeki emojileri ve karmaşık yapıları güvenli algılar."""
    if exit_signal == "SAT":
        return "🔴 SAT (RİSK KORUMA)"
    
    # "AL" veya "🚀 MEGA AL" gibi durumları esnek yakalamak için 'in' eklendi
    if "SAT" in signal and exit_signal == "TUT":
        return "⚪ BEKLE (KARARSIZ)"
        
    return "🟢 POZİSYONU KORU" if "AL" in signal else "⚪ NÖTR"


def _get_empty_result():
    return {
        "price": 0.0, "rsi": 0.0, "adx": 0.0,
        "flow": 0.0, "flow_raw": 0.0, "flow_signal": 0.0,
        "buy_volume": 0.0, "sell_volume": 0.0,
        "buy_pressure": 0.0, "sell_pressure": 0.0,
        "smart_money": 0.0, "guven_skoru": 0.0,
        "rr": 1.0, "score": 0.0, "final_score": 0.0,
        "grade": "N/A", "signal": "⚠️ VERİ YETERSİZ", "color": "#757575",
        "destek": 0.0, "direnc": 0.0,
        "regime": "Bilinmiyor", "trend_text": "Veri Yetersiz", "piyasa_notu": "Yetersiz Veri",
        "stop_loss": 0.0, "take_profit": 0.0,
        "stock_score": 0.0, "hisse_karnesi": 0.0, "flow_index": 0.0,
        "ai_comment": "Teknik analiz ve yapay zeka modellerinin çalışabilmesi için daha fazla geçmiş bar oluşması beklenmelidir.",
        "strategy_action": "BEKLE/SAT",
        "kalicilik_durumu": "⚠️ VERİ YETERSİZ (Hacim Analizi Yapılamadı)",
        "suni_hareket": "⚠️ VERİ YETERSİZ",
        "suni_kod": 0,
        "kar_koruma_skoru": 0, "kar_koruma_durumu": "N/A", "tepe_skoru": 0,
        "tepe_bolgesi_durumu": "N/A", "zirve_yorgunlugu": "N/A", "gun_ici_alarm": "N/A",
        "exit_strategy_action": "BEKLE/SAT", "exit_strategy_comment": "Veri Yetersiz"
    }



def _clean_and_prepare_df(df, model, features, symbol):
    """DataFrame temizliği, sütun zırhı ve yapay zeka hazırlık süreçlerini yönetir."""
    st.markdown("### 🛠️ Arka Plan Veri Denetimi")
    
    if df is None or len(df) < 30:
        st.error("❌ HATA: DataFrame boş veya 30 bardan az veri geliyor!")
        return None, features

    st.info(f"1. Adım: Veri tabanından gelen ham satır sayısı: **{len(df)}**")
    df = df.copy()

    # MULTI-INDEX VE TUPLE SÜTUN ZIRHI
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    cleaned_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            cleaned_cols.append(str(col[0]).lower().strip())
        else:
            cleaned_cols.append(str(col).lower().strip())
    df.columns = cleaned_cols
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        else:
            st.warning(f"⚠️ Kritik sütun '{col}' isimle bulunamadı, pozisyonel eşleştirme yapılıyor.")
            if col == 'open' and len(df.columns) > 0: df.rename(columns={df.columns[0]: 'open'}, inplace=True)
            elif col == 'high' and len(df.columns) > 1: df.rename(columns={df.columns[1]: 'high'}, inplace=True)
            elif col == 'low' and len(df.columns) > 2: df.rename(columns={df.columns[2]: 'low'}, inplace=True)
            elif col == 'close' and len(df.columns) > 3: df.rename(columns={df.columns[3]: 'close'}, inplace=True)

    if 'close' not in df.columns:
        st.error("🚨 KRİTİK HATA: Sütunlar hiçbir şekilde 'close' olarak eşleştirilemedi! Mevcut Sütunlar: " + str(list(df.columns)))
        return None, features
            
    st.write(f"2. Adım: Temizlik sonrası ilk ham kapanış fiyatı: `{df['close'].iloc[-1]}`")

    # PREPARE DATA KONTROLÜ
    try:
        df, features = _prepare_data(df, features, model)
        if df is None or df.empty:
            st.error("❌ HATA: `_prepare_data` fonksiyonu veri setini tamamen sıfırladı/boşalttı!")
            return None, features
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(col).lower().strip() for col in df.columns]
        features = [f.lower() for f in features]
        
        st.success(f"3. Adım: `_prepare_data` sonrası satır sayısı: **{len(df)}**")
        st.write(f"4. Adım: Yapay zeka hazırlığı sonrası fiyat durumu: `{df['close'].iloc[-1]}`")
        
    except Exception as e:
        st.error(f"❌ HATA: `_prepare_data` fonksiyonu çalışırken çöktü! Hata: {e}")

    df = df.ffill().bfill()
    return df, features

def _calculate_indicators_and_flow(df, model, features, symbol):
    """Canlı fiyatı, teknik indikatörleri, ML tahmin olasılıklarını ve para akışını hesaplar."""
    # LIVE PRICE (Canlı Fiyat Entegrasyonu)
    live_price = None
    try:
        live_price = get_live_price(symbol)
    except Exception:
        live_price = None

    if not is_market_open():
        try:
            yf_symbol = f"{symbol}.IS" if not symbol.endswith(".IS") else symbol
            df_daily = yf.download(yf_symbol, period="1d", interval="1d", auto_adjust=True, progress=False)
            if not df_daily.empty:
                if isinstance(df_daily.columns, pd.MultiIndex):
                    df_daily.columns = df_daily.columns.get_level_values(0)
                df_daily.columns = [str(c).lower().strip() for c in df_daily.columns]
                live_price = float(df_daily["close"].iloc[-1])
        except Exception:
            pass

    if live_price and live_price > 0:
        close = float(live_price)
        idx = df.index[-1]
        df.loc[idx, "close"] = close
        
        current_high = float(df.loc[idx, "high"].iloc[0]) if hasattr(df.loc[idx, "high"], "iloc") else float(df.loc[idx, "high"])
        current_low = float(df.loc[idx, "low"].iloc[0]) if hasattr(df.loc[idx, "low"], "iloc") else float(df.loc[idx, "low"])
        
        df.loc[idx, "high"] = max(current_high, close)
        df.loc[idx, "low"] = min(current_low, close)
    else:
        close = float(df["close"].iloc[-1])

    st.metric(label="5. Adım: Analize Giren Net Kapanış Fiyatı", value=f"{close} ₺")
    base_last = df.iloc[-1].copy()

    # BASE_LAST SÜTUN TEMİZLİK ZIRHI
    clean_last_dict = {}
    for col in list(df.columns):
        col_lower = str(col).lower().strip()
        val = base_last[col]
        try:
            import numpy as np
            if hasattr(val, "values"):
                flat_vals = np.asarray(val.values).ravel()
                pure_val = float(flat_vals[0]) if len(flat_vals) > 0 else float(df['close'].iloc[-1])
            else:
                pure_val = float(val)
        except:
            pure_val = float(df['close'].iloc[-1])
            
        clean_last_dict[col_lower] = pure_val
        clean_last_dict[str(col).strip()] = pure_val
    base_last = clean_last_dict

    # MODEL PREDICTION (XGBoost & LightGBM Ensemble)
    proba = 0.5
    try:
        valid_features = [f for f in features if f in df.columns]
        if len(valid_features) == len(features):
            X = df[features].iloc[-1:].values
            probs = []
            xgb_model = model.get("xgb", None) if isinstance(model, dict) else None
            if xgb_model is not None and hasattr(xgb_model, "predict_proba"):
                try: comps = xgb_model.predict_proba(X); probs.append(float(comps[0][1]))
                except Exception: pass

            lgbm_model = model.get("lgbm", None) if isinstance(model, dict) else None
            if lgbm_model is not None and hasattr(lgbm_model, "predict_proba"):
                try: comps = lgbm_model.predict_proba(X); probs.append(float(comps[0][1]))
                except Exception: pass

            if len(probs) > 0:
                proba = sum(probs) / len(probs)
        else:
            st.warning("⚠️ Bazı yapay zeka model özellikleri (features) DataFrame'de eksik!")
    except Exception:
        proba = 0.5

    proba = float(np.clip(proba, 0.05, 0.95))

    # INDICATORS
    try:
        adx_series = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], fillna=True).adx()
        adx = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 25.0
        
        rsi_series = ta.momentum.RSIIndicator(close=df["close"], fillna=True).rsi()
        rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
    except Exception:
        adx = 25.0
        rsi = 50.0

    return df, close, base_last, proba, adx, rsi 


def execute_central_decision_engine(result, score, suni_durum, str_signal, tepe_skoru):
    """
    KUMANDA MERKEZİ V5.1: MOPAS/TUREX karşılaştırmasında tespit edilen statik metin 
    hatasını düzeltir. AI yorum alanını tamamen dinamik ve değişken odaklı hale getirir.
    """
    # 1. PARAMETRE KONTROLLERİ
    try:
        score_val = float(score)
    except (ValueError, TypeError):
        score_val = 50.0

    try:
        tepe_val = float(tepe_skoru)
    except:
        tepe_val = 50.0

    # Flow verisini güvenli oku
    try:
        flow_idx = float(result.get("Flow Index", result.get("flow", result.get("flow_index", 0.0))))
    except:
        flow_idx = 0.0

    # 2. HASSAS GRADING SİSTEMİ
    if score_val >= 80:     current_grade = "A+"
    elif score_val >= 70:   current_grade = "A"
    elif score_val >= 50:   current_grade = "B"
    elif score_val >= 43:   current_grade = "C"
    elif score_val >= 35:   current_grade = "C-"
    elif score_val >= 20:   current_grade = "D"
    else:                   current_grade = "F"

    result["score"] = score_val
    result["grade"] = current_grade
    result["stock_score"] = score_val
    result["hisse_karnesi"] = score_val

    suni_durum = str(suni_durum or "").upper()
    str_signal = str(str_signal or "").upper()

    # 3. DİNAMİK FLOW TANIMLAMASI (Metin hatasını çözen anahtar yer)
    if abs(flow_idx) <= 0.15:
        flow_text = "istatistiksel olarak dengeli bantta yer almakta"
    elif flow_idx < -0.15:
        flow_text = "hafif negatif eğilimli para çıkışı barındırmakta"
    else:
        flow_text = "hafif pozitif eğilimli para girişi barındırmakta"

    is_anomaly = "SUNİ" in suni_durum or "TUZAK" in suni_durum

    # 4. TEK BEYİN REJİMİ
    # DURUM 1: DÜŞÜK ZİRVE RİSKİ BÖLGESİ (Tepe Skoru < 45)
    if tepe_val < 45:
        if is_anomaly:
            result["suni_hareket"] = "🟡 YATAY KONSOLİDASYON"
            result["kalicilik_durumu"] = "🟡 YATAY KONSOLİDASYON"
        
        if "SAT" in str_signal or score_val < 43:
            result["signal"] = "İZLE / NÖTR"
            result["strategy_action"] = "SÜREÇ: YATAY DENGELENME"
            result["kar_koruma_durumu"] = "🟢 NORMAL BÖLGE"
            result["exit_strategy_action"] = "POZİSYONU KORU"
            result["trend_text"] = f"⚖️ Zirve Baskısı Düşük (Mevcut Seviye: %{tepe_val:.0f})"
            
            # Tamamen dinamikleştirilmiş hata vermeyen yeni metin yapısı:
            result["ai_comment"] = (
                f"ℹ️ SENTEZ RAPORU: Fiyat yapısı, zirve riskinin düşük olduğu (%{tepe_val:.0f}) "
                f"yatay bir konsolidasyon ve dengelenme sürecini işaret etmektedir. Para akışı (Flow: {flow_idx:.3f}) "
                f"{flow_text} olup, net bir kırılım iştahı taşımamaktadır. "
                f"Zayıf/Dengeli karne notu ({current_grade}) piyasadaki kararsız yapıyı onaylamakta olup, "
                f"mevcut pozisyonlar açısından nötr/bekleme stratejisi uygundur."
            )
            result["exit_strategy_comment"] = (
                "Yatay konsolidasyon bandının ana destek seviyeleri geçerliliğini korumaktadır. "
                "Hacim veri setinde yönlü bir kırılım oluşana kadar mevcut risk limitleri dahilinde pozisyon takibi önerilir."
            )
        else:
            result["signal"] = "AL / KADEMELİ"
            result["strategy_action"] = "YATAY BANTTAN ÇIKIŞ ÇABASI"
            result["kar_koruma_durumu"] = "🟢 GÜVENLİ BÖLGE"
            result["exit_strategy_action"] = "DESTEK ÜZERİ İZLEME"
            result["trend_text"] = "🟢 Pozitif Akümülasyon"

    # DURUM 2: YÜKSEK ZİRVE RİSKİ BÖLGESİ (Tepe Skoru >= 45)
    else:
        if is_anomaly or "SAT" in str_signal or score_val < 45:
            if is_anomaly:
                result["suni_hareket"] = "⚠️ SUNİ ZİRVE YÜKSELİŞİ"
                result["kalicilik_durumu"] = "⚠️ SUNİ ZİRVE YÜKSELİŞİ"
            
            result["signal"] = "UZAK DUR / SAT"
            result["strategy_action"] = "ZİRVE BÖLGESİ BOĞA TUZAĞI"
            result["kar_koruma_durumu"] = "🔴 DEFANSİF MOD (ALARM)"
            result["exit_strategy_action"] = "KADEMELİ AZALT / NAKİTE GEÇ"
            result["trend_text"] = f"🚨 Yüksek Zirve Riski (Mevcut Seviye: %{tepe_val:.0f})"
            
            result["ai_comment"] = (
                f"🚨 DİKKAT: Hisse zirve direnç hattına yakın, yüksek risk bölgesindedir (%{tepe_val:.0f}). "
                f"Mevcut fiyat hareketinin hacim grubuyla desteklenmemesi ve karne notunun ({current_grade}) "
                f"bu seviyede zayıf kalması, zirve bölgesinde bir boğa tuzağı olasılığını artırmaktadır."
            )
            result["exit_strategy_comment"] = (
                "Direnç seviyelerindeki hacimsiz seyir sebebiyle yeni maliyetlenme yapılmamalı, "
                "olası kar realizasyonlarına karşı iz süren stoplar koruma amaçlı yukarı çekilmelidir."
            )
        else:
            result["suni_hareket"] = "🟢 HACİMLİ TREND"
            result["kalicilik_durumu"] = "🟢 HACİMLİ TREND"
            result["signal"] = "AL / TRENDİ SÜR"
            result["strategy_action"] = "TREND DEVAM FORMASYONU"
            result["kar_koruma_durumu"] = "🟢 POZİTİF İZLEME"
            result["exit_strategy_action"] = "İZ SÜREN STOPLA TAKİP"
            result["trend_text"] = "🚀 Güçlü Trend Devamı"

    return result

def analyze(df, model, features, symbol="THYAO"):
    # -------------------------------------------------------------------------
    # ADIM 1: VERİ TEMİZLİK VE KONTROL
    # -------------------------------------------------------------------------
    with st.expander(f"🔍 {symbol} Veri Akış ve Teşhis Raporu", expanded=False):
        df, features = _clean_and_prepare_df(df, model, features, symbol)
        if df is None:
            return _get_empty_result()

        # -------------------------------------------------------------------------
        # ADIM 2: CANLI FİYAT VE TEKNİK İNDİKATÖRLERİN HESAPLANMASI
        # -------------------------------------------------------------------------
        df, close, base_last, proba, adx, rsi = _calculate_indicators_and_flow(df, model, features, symbol)

    # -------------------------------------------------------------------------
    # ADIM 3: SÜTUN UYUMSUZLUK KÖPRÜSÜ (BÜYÜK/KÜÇÜK HARF GÜVENLİĞİ)
    # -------------------------------------------------------------------------
    df = df.loc[:, ~df.columns.duplicated()]
    existing_cols = list(df.columns)
    for col in existing_cols:
        cap_col = str(col).capitalize().strip()
        low_col = str(col).lower().strip()
        if cap_col not in df.columns: df[cap_col] = df[col]
        if low_col not in df.columns: df[low_col] = df[col]

    # -------------------------------------------------------------------------
    # ADIM 4: FLOW, SMART MONEY VE SİNYAL MOTORLARININ TETİKLENMESI
    # -------------------------------------------------------------------------
    flow = get_hybrid_flow(symbol, df)
    flow_index = float(flow.get("flow_index", 0.0))
    smart_money = float(calculate_hybrid_smart_money(symbol, close, flow.get("flow_signal", flow_index), last=base_last))

    smart_score = (proba * 100 * 0.4 + (flow_index + 1) * 20 + (50 - abs(rsi - 50)) * 0.4)
    smart_score = float(np.clip(smart_score, 0, 100))

    engine = institutional_signal_engine(
        df=df, adx=adx, rsi=rsi,
        flow_signal=flow.get("flow_signal", 0),
        buy_pressure=flow.get("buy_pressure", 0.0),
        sell_pressure=flow.get("sell_pressure", 0.0),
        close=close, flow_index=flow_index, smart_money=smart_money
    )

    score_raw = engine.get("skor", 50.0)
    signal = engine.get("sinyal", "BEKLE").replace("NEUTRAL", "BEKLE").replace("⚪ BEKLE", "BEKLE")
    color = "#90A4AE"

    # FINAL METRİKLER VE STRATEJİ HESAPLAMALARI
    stock_score = float(np.clip(score_raw * 0.40 + smart_money * 0.30, 0, 100))
    final_score = float(np.clip(score_raw * 0.60 + smart_money * 0.25, 0, 100))

    destek, direnc, rr, risk, reward, piyasa_notu = _calculate_risk_metrics(
        df, close, bool(df["close"].tail(20).nunique() <= 1)
    )

    gunluk_getiri = 0.0
    if len(df) >= 2:
        prev_close = float(df["close"].iloc[-2])
        if prev_close > 0: gunluk_getiri = ((close - prev_close) / prev_close) * 100

    exit_data = calculate_exit_strategy(
        price=close, rsi=rsi, buy_pressure=float(flow.get("buy_pressure", 0.0)),
        flow_signal=float(flow.get("flow_signal", flow_index)),
        ma50=float(base_last.get("ma50", close)),
        gunluk_getiri=gunluk_getiri, direnc=float(direnc), destek=float(destek)
    )

    if 'macd' not in df.columns:
        df['macd'] = 0.0
        df['macd_signal'] = 0.0
        df['macd_hist'] = 0.0

    # -------------------------------------------------------------------------
    # ADIM 5: ANA SONUÇ SÖZLÜĞÜNÜN İNŞASI
    # -------------------------------------------------------------------------
    result = {
        "price": close, "rsi": rsi, "adx": adx, "flow_index": flow_index, "flow": flow_index,
        "flow_raw": flow.get("flow_raw", 0.0), "flow_signal": flow.get("flow_signal", 0.0),
        "buy_pressure": flow.get("buy_pressure", 50.0), "sell_pressure": flow.get("sell_pressure", 50.0),
        "buy_volume": float(flow.get("buy_volume", 0.0)), "sell_volume": float(flow.get("sell_volume", 0.0)),
        "smart_money": smart_money, "smart_score": smart_score, "smart_signal": signal,
        "smart_confidence": engine.get("güven", 50), "market_regime": engine.get("rejim", "Normal"),
        "regime": engine.get("rejim", "Normal"), "trend_text": engine.get("trend_text", "Dengeli"),
        "divergence": engine.get("uyumsuzluk", "YOK"), "score": stock_score, "final_score": final_score,
        "stock_score": stock_score, "hisse_karnesi": stock_score, "grade": "C", "color": color, "signal": signal,
        "destek": destek, "direnc": direnc, "rr": rr, "piyasa_notu": piyasa_notu if piyasa_notu else "Normal Rejim",
        "trend_power": engine.get("trend_gucu", 0),
        "stop_loss": exit_data.get("stop_loss", 0.0), "take_profit": exit_data.get("take_profit", 0.0),
        "suni_hareket": flow.get("kalicilik_durumu", "🟢 NORMAL"), "kalicilik_durumu": flow.get("kalicilik_durumu", "🟢 NORMAL"),
        "suni_kod": flow.get("suni_kod", 0),
        "strategy_action": "AL" if "AL" in signal else "SAT" if "SAT" in signal else "BEKLE",
        "exit_strategy_action": exit_data.get("exit_strategy_action", "BEKLE"),
        "exit_strategy_comment": exit_data.get("exit_strategy_comment", ""), "exit_strategy": exit_data,
        "guven_skoru": engine.get("güven", 50), "ai_comment": "Analiz tamamlandı.",
        "kar_koruma_skoru": exit_data.get("kar_koruma_skoru", 0), "kar_koruma_durumu": exit_data.get("kar_koruma_durumu", "N/A"),
        "tepe_skoru": exit_data.get("tepe_skoru", 0), "tepe_bolgesi_durumu": exit_data.get("tepe_bolgesi_durumu", "N/A"),
        "zirve_yorgunlugu": exit_data.get("zirve_yorgunlugu", "N/A"), "gun_ici_alarm": exit_data.get("gun_ici_alarm", "N/A")
    }

    # -------------------------------------------------------------------------
    # ADIM 6: MERKEZİ ANANAYASA FİLTRESİNİN ÇALIŞTIRILMASI
    # -------------------------------------------------------------------------
    result = execute_central_decision_engine(
        result=result,
        score=stock_score,
        suni_durum=result.get("suni_hareket", ""),
        str_signal=result.get("signal", ""),
        tepe_skoru=result.get("tepe_skoru", 25)
    )

    # -------------------------------------------------------------------------
    # ADIM 7: R/R EKRAN FORMATLAMA VE ÇIKIŞ
    # -------------------------------------------------------------------------
    final_rr_val = result.get('rr_ratio', result.get('R/R', result.get('rr', 2.67)))
    try:
        st.metric(label="📊 Getiri / Risk Oranı (R/R)", value=f"{float(final_rr_val):.2f}")
    except (ValueError, TypeError):
        st.metric(label="📊 Getiri / Risk Oranı (R/R)", value=f"{final_rr_val}")

    return result



def create_technical_chart(df):
    """Teknik analiz grafiğini güvenli ve harf duyarsız şekilde oluşturur."""
    import pandas as pd
    
    if df is None or df.empty:
        return None

    # 🌟 GRAFİK KİLİTLENMESİNİ ÖNLEN KISIM:
    df_chart = df.copy()
    # --- GÜVENLİK DUVARI: Eksik sütunları 0'la doldur ---
    required_cols = ['macd', 'macd_signal', 'macd_hist', 'ma50', 'ma200', 'rsi']
    for col in required_cols:
        if col not in df_chart.columns:
            df_chart[col] = 0.0
    
    # Mükerrer sütunlar varsa hemen temizle
    df_chart = df_chart.loc[:, ~df_chart.columns.duplicated()]
    
    # Eğer yfinance kaynaklı Multi-Index kalmışsa temizle
    if isinstance(df_chart.columns, pd.MultiIndex):
        df_chart.columns = df_chart.columns.get_level_values(0)
        
    # Sütunları küçük harfe çekerek standartlaştırıyoruz
    df_chart.columns = [str(col).lower().strip() for col in df_chart.columns]

    # Fonksiyonun içindeki tüm df_chart["Close"], df_chart["Open"] vb. çağrılarını 
    # güvenle karşılasın diye büyük harfli klonlarını oluşturuyoruz
    df_chart["Close"] = pd.to_numeric(df_chart["close"], errors="coerce")
    if "open" in df_chart.columns: df_chart["Open"] = pd.to_numeric(df_chart["open"], errors="coerce")
    if "high" in df_chart.columns: df_chart["High"] = pd.to_numeric(df_chart["high"], errors="coerce")
    if "low" in df_chart.columns: df_chart["Low"] = pd.to_numeric(df_chart["low"], errors="coerce")
    if "volume" in df_chart.columns: df_chart["Volume"] = pd.to_numeric(df_chart["volume"], errors="coerce")
    
    df_chart["MACD"] = pd.to_numeric(df_chart.get("macd", 0), errors="coerce")
    df_chart["MACD_Signal"] = pd.to_numeric(df_chart.get("macd_signal", 0), errors="coerce")
    df_chart["MACD_Hist"] = pd.to_numeric(df_chart.get("macd_hist", 0), errors="coerce")
    
    # Boşlukları doldur ki grafik çizgilerinde kopukluk olmasın
    df_chart = df_chart.ffill().bfill()
    # Hareketli Ortalamalar (MA) Kontrolü (Yeni halka arzlar için min_periods eklendi)
    if "ma50" not in df_chart.columns: 
        df_chart["ma50"] = df_chart["Close"].rolling(50, min_periods=1).mean()
        
    if "ma200" not in df_chart.columns: 
        df_chart["ma200"] = df_chart["Close"].rolling(200, min_periods=1).mean()

    # RSI Kontrolü
    if "rsi" not in df_chart.columns: 
        df_chart["rsi"] = 50

    # --- 2. 4 PANELLİ SUBPLOT YAPISI VE ORANLAMASI ---
    # Plotly'de row_width dizisi EN ALT PANELİN (Row 4) yüksekliğinden başlayarak yukarı doğru sıralanır!
    # [Row 4 Yüksekliği, Row 3 Yüksekliği, Row 2 Yüksekliği, Row 1 Yüksekliği]
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.04,  # Paneller arası boşluk oranlandı
        row_width=[0.20, 0.15, 0.15, 0.50]  # Sıralama düzeltildi: MACD(%20), RSI(%15), Hacim(%15), Mumlar(%50)
    )

    # --- PANEL 1: ANA MUM GRAFİĞİ VE MA ÇİZGİLERİ ---
    fig.add_trace(
        go.Candlestick(
            x=df_chart.index,
            open=df_chart['Open'],
            high=df_chart['High'],
            low=df_chart['Low'],
            close=df_chart['Close'],
            name="Fiyat (Candle)",
            increasing_line_color='#00E676', # Canlı Yeşil
            decreasing_line_color='#FF5252'  # Canlı Kırmızı
        ), row=1, col=1
    )
    
    # MA 50 Çizgisi
    fig.add_trace(
        go.Scatter(
            x=df_chart.index, y=df_chart['ma50'],
            line=dict(color='#29B6F6', width=1.5), # Mavi
            name="MA 50"
        ), row=1, col=1
    )
    
    # MA 200 Çizgisi
    fig.add_trace(
        go.Scatter(
            x=df_chart.index, y=df_chart['ma200'],
            line=dict(color='#FFCA28', width=2), # Altın Sarısı
            name="MA 200"
        ), row=1, col=1
    )

    # --- PANEL 2: HACİM (VOLUME) ---
    volume_colors = [
        '#00E676' if close >= open_val else '#FF5252' 
        for open_val, close in zip(df_chart['Open'], df_chart['Close'])
    ]
    
    fig.add_trace(
        go.Bar(
            x=df_chart.index, y=df_chart['Volume'],
            marker_color=volume_colors,
            opacity=0.7,
            name="Hacim"
        ), row=2, col=1
    )

    # --- PANEL 3: RSI ---
    fig.add_trace(
        go.Scatter(
            x=df_chart.index, y=df_chart['rsi'],
            line=dict(color='#E040FB', width=2), # Mor
            name="RSI (14)"
        ), row=3, col=1
    )
    
    # RSI 30 ve 70 Referans Çizgileri
    fig.add_shape(type="line", x0=df_chart.index[0], x1=df_chart.index[-1], y0=70, y1=70,
                  line=dict(color="rgba(255, 82, 82, 0.4)", width=1, dash="dash"), row=3, col=1)
    fig.add_shape(type="line", x0=df_chart.index[0], x1=df_chart.index[-1], y0=30, y1=30,
                  line=dict(color="rgba(0, 230, 118, 0.4)", width=1, dash="dash"), row=3, col=1)

    # --- PANEL 4: MACD ---
    # MACD Çizgisi
    fig.add_trace(
        go.Scatter(
            x=df_chart.index, y=df_chart['macd'],
            line=dict(color='#29B6F6', width=1.5),
            name="MACD"
        ), row=4, col=1
    )
    
    # Sinyal Çizgisi
    fig.add_trace(
        go.Scatter(
            x=df_chart.index, y=df_chart['macd_signal'],
            line=dict(color='#FF9100', width=1.5), # Turuncu
            name="Sinyal"
        ), row=4, col=1
    )
    
    # MACD Histogramı
    hist_colors = ['#00E676' if val >= 0 else '#FF5252' for val in df_chart['macd_hist']]
    fig.add_trace(
        go.Bar(
            x=df_chart.index, y=df_chart['macd_hist'],
            marker_color=hist_colors,
            name="Histogram"
        ), row=4, col=1
    )

    # --- 3. GRAFİK YÖNETİMİ VE STİL AYARLARI ---
    fig.update_layout(
        template="plotly_dark", 
        height=800,             # Mobil ve masaüstü dengeli dikey boyut
        margin=dict(l=55, r=20, t=30, b=30),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"    # Dikey imleç takibi aktif
    )

    # Dikey Eksen Başlıkları
    fig.update_yaxes(title_text="Fiyat", row=1, col=1)
    fig.update_yaxes(title_text="Hacim", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    # CRITICAL FIX: Range slider'ı ve tarih çakışmalarını tüm alt panellerde kapatıyoruz
    fig.update_xaxes(rangeslider_visible=False)

    return fig


    # -------------------------------------------------------------------------
    # Bundan sonrası senin mevcut grafik çizim kodların (Aynen devam ediyor):
    # Örn: df_chart["ma50"] = df_chart["Close"].rolling(50, min_periods=1).mean()
    # -------------------------------------------------------------------------


# =============================================================
# 🛠️ GÜVENLİ DÖNÜŞTÜRÜCÜ FONKSİYONLAR
# =============================================================
def safe_float(val):
    """Null/None korumalı float dönüştürücü."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

def safe_str(val, default="N/A"):
    """Null/None korumalı string dönüştürücü."""
    return str(val) if val is not None else default



# ==============================================================================
# 4. STREAMLIT MASAÜSTÜ UYGULAMA PANELİ (MANTIK VE CSS KORUNDU)
# ==============================================================================
if IS_STREAMLIT:    
    st.set_page_config(page_title="Masaüstü Borsa", layout="wide")
    
    # --- CSS PANEL (GÜVENLİ VE MOBİL UYUMLU) ---
    st.markdown("""
        <style>
        /* Ana Arka Plan ve Yazı Rengi */
        .stApp { 
            background-color: #121212; 
            color: #FFFFFF; 
        }
        
        /* Expander İç Alan Tasarımları */
        div[data-testid="stExpander"] { 
            background-color: #1E1E1E; 
            border: 1px solid #2D2D2D; 
            border-radius: 10px; 
        }
        
        /* Form Gönderim Butonları */
        div.stFormSubmitButton > button { 
            background-color: #007BFF !important; 
            color: white !important; 
            width: 100% !important; 
            border-radius: 8px !important;
        }
        
        /* GENEL BUTON TASARIMI */
        div.stButton > button { 
            background-color: #007BFF !important; 
            color: white !important;
            font-size: 14px !important; 
            white-space: nowrap !important; 
            padding: 10px 16px !important;
            font-weight: bold !important;
            border-radius: 8px !important;
            border: none !important;
            transition: background-color 0.2s ease;
        }
        div.stButton > button:hover {
            background-color: #0056b3 !important;
        }
        
        /* Üst Boşluk Optimizasyonu */
        [data-testid="stMainBlockContainer"] {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* Kaydırılabilir Tablo/Liste Alanı */
        .scrollable-container {
            max-height: 380px !important;
            overflow-y: auto !important;
            padding-right: 5px;
        }
        
        /* Özel Silme Butonu Rengi */
        div.stButton > button[key^="global_delete_btn"] { 
            background-color: #E74C3C !important; 
            color: white !important; 
        }
        div.stButton > button[key^="global_delete_btn"]:hover { 
            background-color: #c0392b !important; 
        }

        /* Checkbox Hizalaması */
        div[data-testid="stCheckbox"] { 
            margin-top: 8px !important; 
        }

        /* Popover Tasarımları */
        div[data-testid="stPopover"] button { 
            background-color: #2D2D2D !important;
            border: 1px solid #444444 !important; 
            color: #00F0FF !important;
            border-radius: 6px !important;
        }
        div[data-testid="stPopoverBody"] button { 
            background: none !important; 
            color: white !important; 
            text-align: left !important; 
            width: 100% !important; 
        }
        div[data-testid="stPopoverBody"] button:hover { 
            background-color: #007BFF !important; 
        }
        </style>
    """, unsafe_allow_html=True)

    tr_zamanı = datetime.now(ZoneInfo("Europe/Istanbul")).strftime('%H:%M:%S')
    st.title("🖥️ Borsa")
    st.caption(f"⏱️ Canlı takip tablosu 60 saniyede bir otomatik güncellenir. Son Yenilenme: {tr_zamanı}")
    
   
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

    # 5 Adet Gelişmiş Sekme Yapısı
    sekme1, sekme2, sekme3, sekme4, sekme5 = st.tabs(["PORTFÖY & STOP", "HİSSE ANALİZ", "MEGA RADAR", "Mega 2", "YAPAY ZEKA"])

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
    
    with sekme2:
        # Otomatik yenileme motoru (60 saniye)
        st_autorefresh(interval=60000, key="refresh_sekme2")

        if "aktif_hisse" not in st.session_state:
            st.session_state.aktif_hisse = ""

        # =====================================================================
        # 1. AKILLI ARAMA FORMU
        # =====================================================================
        with st.form(key="hisse_arama_formu", clear_on_submit=True):
            hisse_input = st.text_input("Hisse Senedi Kodu (Örn: THYAO, TUREX):")
            submit_button = st.form_submit_button("🔍 Analiz Et", use_container_width=False)
            
            if submit_button and hisse_input:
                st.session_state.aktif_hisse = hisse_input.strip().upper()

        # =====================================================================
        # 2. ANA ÇALIŞMA BLOKU
        # =====================================================================
        if st.session_state.aktif_hisse:
            hisse_kodu = st.session_state.aktif_hisse
            
            if hisse_kodu.endswith(".IS"):
                hisse_temiz = hisse_kodu.replace(".IS", "")
            else:
                hisse_temiz = hisse_kodu
                
            sembol = f"{hisse_temiz}.IS"
            st.caption(f"🔎 Şu an incelenen hisse: **{sembol}**")

            market_active = is_market_open(strict=True)

            if market_active:
                p_time, i_time = "5d", "5m"
            else:
                p_time, i_time = "1y", "1d"
                
            df_st = get_live_data(sembol, period=p_time, interval=i_time)

            if df_st is None or df_st.empty:
                df_st = get_live_data(hisse_temiz, period=p_time, interval=i_time)
                if df_st is not None and not df_st.empty:
                    sembol = hisse_temiz

            model_st = load_model()

            if df_st is None or df_st.empty:
                st.error(f"❌ {sembol} için veri çekilemedi. Lütfen kodu veya internet bağlantınızı kontrol edin.")
            elif model_st is None:
                st.error("❌ Yapay zeka modeli sistemden yüklenemedi.")
            else:
                result = analyze(df_st, model_st, MODEL_FEATURES, symbol=hisse_temiz)
                
                if not result:
                    st.warning("⚠️ Analiz motoru bu hisse için sonuç üretemedi.")
                else:
                    # Zaman Damgası Paneli
                    su_an = datetime.now(ZoneInfo("Europe/Istanbul")).strftime("%H:%M:%S")
                    st.markdown(f"""
                    <div style="background-color: #1E1E1E; padding: 10px; border-radius: 8px; border-left: 5px solid #2196f3; margin-bottom: 20px;">
                        ⏱️ <b>Son Güncellenme Zamanı:</b> <span style="color: #2196f3; font-weight: bold;">{su_an}</span> 
                        <span style="font-size: 12px; color: #888888; margin-left: 10px;">(Otomatik Takip Aktif)</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 🌟 PERFORMANCE GUARD: Veri setini mobil/grafik performansı için kırpıyoruz
                    # iPhone ekranında zaten 60'tan fazla bar yan yana sığmaz, RAM'i korur.
                    if df_st is not None and not df_st.empty:
                        df_chart_data = df_st.tail(60) if len(df_st) > 60 else df_st
                    else:
                        df_chart_data = df_st
                    
                    # Kalıcılık ve Tuzak Dedektörü (HTML hafifletildi)
                    suni_kod = result.get("suni_kod", 0)
                    status_color = "#4CAF50" if suni_kod == 2 else ("#FF9800" if suni_kod in [1, -1] else "#90A4AE")
                    status_bg = "rgba(76, 175, 80, 0.05)" if suni_kod == 2 else ("rgba(255, 152, 0, 0.05)" if suni_kod in [1, -1] else "rgba(144, 164, 174, 0.05)")

                    st.markdown(f"""
                    <div style="background-color: {status_bg}; padding: 12px; border-radius: 8px; border: 1px solid {status_color}; border-left: 5px solid {status_color}; margin-bottom: 15px;">
                        <span style="font-size: 12px; color: #888888; text-transform: uppercase; font-weight: bold; letter-spacing: 1px;">Hacim ve Trend Doğrulama</span>
                        <div style="font-size: 18px; font-weight: 700; color: white; margin-top: 3px;">
                            {result.get('kalicilik_durumu', 'Veri Yok')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Erken Çıkış & Tepe Analizi
                    tepe_skoru = result.get("tepe_skoru", 0)
                    exit_color = "#4CAF50" if tepe_skoru <= 25 else ("#FFEB3B" if tepe_skoru <= 50 else ("#FF9800" if tepe_skoru <= 75 else "#F44336"))
                    tavsiye_aksiyon = result.get('exit_strategy_action', result.get('exit_action', result.get('strategy_action', 'TUT')))

                    st.markdown(f"""
                    <div style="background-color: #141414; padding: 14px; border-radius: 8px; border: 1px solid #222222; border-top: 4px solid {exit_color}; margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 12px; color: #AAAAAA; font-weight: bold;">🎯 ERKEN ÇIKIŞ & TEPE ANALİZİ</span>
                            <span style="background-color: {exit_color}; color: #000000; padding: 1px 6px; border-radius: 10px; font-size: 11px; font-weight: 800;">
                                TEPE SKORU: %{tepe_skoru}
                            </span>
                        </div>
                        <div style="margin-top: 8px; font-size: 13px; color: #DDD;">
                            <p style="margin: 2px 0;"><b>Tepe Bölgesi Durumu:</b> {result.get('tepe_bolgesi_durumu', 'Normal')}</p>
                            <p style="margin: 2px 0;"><b>Kâr Koruma Safhası:</b> {result.get('kar_koruma_durumu', 'Tut')}</p>
                            <p style="margin: 2px 0;"><b>Zirve Yorgunluğu:</b> {result.get('zirve_yorgunlugu', 'Normal')}</p>
                            <p style="margin: 2px 0;"><b>Gün İçi Ekstra Alarm:</b> {result.get('gun_ici_alarm', 'Normal')}</p>
                        </div>
                        <div style="margin-top: 8px; padding-top: 6px; border-top: 1px dashed #333; font-size: 14px; font-weight: bold; color: {exit_color};">
                            👉 Çıkış Aksiyon Tavsiyesi: {tavsiye_aksiyon}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # =============================================================
                    # 📊 KPI PANELİ
                    # =============================================================
                    st.markdown("## 📊 KPI Paneli")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("💵 Fiyat", f"{safe_float(result.get('price')):.2f}")
                    c2.metric("📈 RSI", f"{safe_float(result.get('rsi')):.1f}")
                    c3.metric("📉 ADX", f"{safe_float(result.get('adx')):.1f}")

                    c4, c5, c6 = st.columns(3)
                    flow_percentage = safe_float(result.get('flow')) * 100
                    c4.metric("🌊 Flow", f"%{flow_percentage:.2f}")
                    c5.metric("💼 Smart", f"{safe_float(result.get('smart_money')):.0f}/100")
                    c6.metric("🛡️ Güven", f"%{safe_float(result.get('guven_skoru')):.0f}")

                    c7, c8, c9 = st.columns(3)
                    c7.metric("⚖️ R/R", f"{safe_float(result.get('rr')):.2f}")
                    c8.metric("🎯 Skor", f"{safe_float(result.get('score')):.2f}")
                    c9.metric("🏷️ Grade", safe_str(result.get("grade")))

                    # =============================================================
                    # ARZ / TALEP HACİM PANELİ
                    # =============================================================
                    st.markdown("## 📊 Arz / Talep Analizi")
                    v_col1, v_col2, v_col3 = st.columns(3)
                    # Mobilde taşmayı ve UI sıkışmasını önlemek için Milyon ₺ formatına çektik
                    v_col1.metric("Alım Hacmi", f"{safe_float(result.get('buy_volume'))/1e6:.1f}M ₺")
                    v_col2.metric("Satım Hacmi", f"{safe_float(result.get('sell_volume'))/1e6:.1f}M ₺")
                    v_col3.metric("Flow Index", f"{safe_float(result.get('flow')):.3f}")

                    st.markdown("---")
                    st.markdown(f"""
                    🟢 **Alıcı Baskısı:** %{safe_float(result.get('buy_pressure'))*100:.1f}  
                    🔴 **Satıcı Baskısı:** %{safe_float(result.get('sell_pressure'))*100:.1f}
                    """)
                    
                    col_extra1, col_extra2 = st.columns(2)
                    col_extra1.metric("Suni Hareket Analizi", result.get("suni_hareket", "🟢 NORMAL"))
                    trend_val = result.get("trend_power", 0.0)
                    col_extra2.metric("📈 Trend Gücü", f"{trend_val:.0f}/100")

                    st.warning(f"Ana Strateji: {result.get('strategy_action', 'TUT')} | Pozisyon Yönetimi: {result.get('exit_strategy_action', 'TUT')}")

                    # =============================================================
                    # SIGNAL BOX
                    # =============================================================
                    st.markdown("## 🚦 Sinyal")
                    st.markdown(f"""
                    <div style="background:{result.get('color', '#333')}; padding:14px; border-radius:8px; text-align:center; font-size:24px; font-weight:800; color:white;">
                        {result.get('signal', 'SİNYAL YOK')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # =============================================================
                    # AI STRATEJİ VE PİYASA REJİMİ
                    # =============================================================
                    st.markdown("### 🤖 Yapay Zeka Strateji Raporu")
                    st.info(result.get('ai_comment', 'Yorum bulunamadı.'))
                    st.markdown(f"🎯 **Stratejik Aksiyon:** `{result.get('strategy_action', 'TUT')}`")
                    st.markdown("---")
                    
                    st.markdown("## 📌 Piyasa Rejimi")
                    st.markdown(f"""
                    🧭 **Rejim:** {result.get('regime', 'Bilinmiyor')}  
                    📉 **Destek:** {safe_float(result.get('destek')):.2f}  
                    📈 **Direnç:** {safe_float(result.get('direnc')):.2f}
                    """)
                    
                    # =============================================================
                    # PLOTLY TEKNİK GRAFİK (DARBOĞAZ ÇÖZÜM NOKTASI)
                    # =============================================================
                    st.markdown("## 📈 Teknik Grafik")
                    if df_chart_data is not None and not df_chart_data.empty:
                        # Grafik çizimi için kırpılmış veriyi (df_chart_data) gönderiyoruz
                        fig = create_technical_chart(df_chart_data)
                        if fig:
                            # iPhone Safari çökmesini önleyen özel Plotly konfigürasyonu
                            st.plotly_chart(fig, use_container_width=True, config={
                                'displayModeBar': False,  # Ağır buton menüsünü gizler
                                'responsive': True,
                                'scrollZoom': False       # Yanlışlıkla zoom ile kasmayı önler
                            })
                    else:
                        st.info("Grafik çizimi için veri yetersiz.")

                    # =============================================================
                    # TEKNİK ÖZET VE VERİ KONTROL PANELİ
                    # =============================================================
                    st.markdown("## 📋 Teknik Özet")
                    st.write(f"• Trend Durumu: {result.get('trend_text', 'Bilinmiyor')}")
                    st.write(f"• Stop Loss: {safe_float(result.get('stop_loss')):.2f}")
                    st.write(f"• Kar Al: {safe_float(result.get('take_profit')):.2f}")
                    st.write(f"• Hisse Karnesi: {safe_float(result.get('stock_score')):.0f}/100")
                    st.write(f"• Risk/Getiri Oranı: {safe_float(result.get('rr')):.2f}")

                    if df_st is not None and not df_st.empty:
                        st.write("DATA AGE (SON BAR):", df_st.index[-1])
                        st.write("CURRENT ROW COUNT:", len(df_st))
                        
                        last_close_val = None
                        target_col = next((col for col in ["close", "Close"] if col in df_st.columns), None)
                        if target_col and not df_st.empty:
                            try:
                                val = df_st[target_col].iloc[-1]
                                last_close_val = float(val.values[0]) if hasattr(val, "values") else float(val)
                            except (ValueError, TypeError):
                                last_close_val = None
                                
                        if last_close_val is None or pd.isna(last_close_val):
                            last_close_val = safe_float(result.get('price', 0.0))
                        st.write("LAST CLOSE:", f"{last_close_val:.2f} ₺")

                    if st.checkbox("Debug Modu"):
                        st.json(result)

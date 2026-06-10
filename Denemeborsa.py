# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 22:57:52 2026

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
from zoneinfo import ZoneInfo


# =============================================================================
# 3. ORTAM TESPİTİ VE GÜVENLİK MUHAFIZLARI (CRITICAL FIX)
# =============================================================================
IS_STREAMLIT = False

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is not None:
        IS_STREAMLIT = True
except ImportError:
    pass

# =============================================================================
# 5. AYARLAR VE SUNUCU UYUMLU DİNAMİK DOSYA YOLLARI
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

BASE_DIR = Path(__file__).resolve().parents[0]

def get_model_path():
    if os.path.exists(r"d:\borsa\models\ensemble.pkl"):
        return Path(r"d:\borsa\models\ensemble.pkl")
    return BASE_DIR / "models" / "ensemble.pkl"

MODEL_PATH = get_model_path()

# ==============================================================
# VERI ÇEKME FONKSİYONLARI (BOYUT HATASI KÖKTEN ÇÖZÜLDÜ)
# ==============================================================
# =============================================================================
# VERİ YÖNETİM KATMANI - REFACTOR SAFE
# =============================================================================

def _clean_yfinance_df(df: pd.DataFrame) -> pd.DataFrame:
    """Yahoo Finance verisini analiz motoru için standart hale getirir.

    - MultiIndex kolonları temizler
    - Yinelenen kolonları kaldırır
    - OHLCV kontrolü yapar
    - Sayısal alanları float'a dönüştürür
    - Boş fiyat satırlarını temizler
    - Tarihe göre sıralar
    - Her zaman DataFrame döndürür
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    try:
        # =====================================================================
        # 1. MULTIINDEX TEMİZLİĞİ
        # =====================================================================
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # =====================================================================
        # 2. DUPLICATE SÜTUN TEMİZLİĞİ
        # =====================================================================
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # =====================================================================
        # 3. ZORUNLU OHLCV KONTROLÜ
        # =====================================================================
        required_cols = {"Open", "High", "Low", "Close", "Volume"}

        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            print(f"⚠️ [_clean_yfinance_df] Eksik kolonlar: {missing}")
            return pd.DataFrame()

        # =====================================================================
        # 4. SAYISAL DÖNÜŞÜM
        # =====================================================================
        numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # =====================================================================
        # 5. CLOSE BOŞ OLAN SATIRLARI TEMİZLE
        # =====================================================================
        df = df.dropna(subset=["Close"])

        # =====================================================================
        # 6. INDEX SIRALAMA
        # =====================================================================
        if len(df.index) > 1:
            df = df.sort_index()

        # =====================================================================
        # 7. SON KONTROL
        # =====================================================================
        if df.empty:
            return pd.DataFrame()

        return df

    except Exception as e:
        print(f"⚠️ [_clean_yfinance_df] Hata: {e}")
        return pd.DataFrame()

if IS_STREAMLIT:
    # -------------------------------------------------------------------------
    # STREAMLIT AKTİF ORTAM (CACHE DESTEKLİ)
    # -------------------------------------------------------------------------
    
    @st.cache_data(ttl=60)
    def get_live_data(symbol: str, period="1d", interval="5m"):
        """Anlık takip için 60 saniye ömürlü, kısa periyotlu veri çeker."""
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
        return _clean_yfinance_df(df)

    @st.cache_data(ttl=3600)
    def get_history(symbol: str, period="300d", interval="1d"):
        """Stratejik analizler için 1 saat ömürlü, geniş periyotlu veri çeker."""
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
        return _clean_yfinance_df(df)

else:
    # -------------------------------------------------------------------------
    # ARKA PLAN ORTAMI (CORE / BACKTEST / SCRIPT)
    # -------------------------------------------------------------------------
    
    def get_live_data(symbol: str, period="1d", interval="5m"):
        """Önbelleksiz, doğrudan canlı tahta verisi çeker."""
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
        return _clean_yfinance_df(df)

    def get_history(symbol: str, period="300d", interval="1d"):
        """Önbelleksiz, doğrudan tarihsel bar verisi çeker."""
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
        return _clean_yfinance_df(df)
    
@st.cache_resource(show_spinner=False)
def load_model() -> dict | None:
    """Eğitilmiş modeli yükler ve Streamlit cache'inde saklar.

    Returns:
        dict | None: Model sözlüğü veya hata durumunda None
    """
    if not os.path.isfile(MODEL_PATH):
        print(f"❌ [load_model] Model dosyası bulunamadı: {MODEL_PATH}")
        return None

    try:
        model = joblib.load(MODEL_PATH)

        if model is None:
            print("❌ [load_model] Model boş yüklendi.")
            return None

        if not isinstance(model, dict):
            print(f"❌ [load_model] Geçersiz model tipi: {type(model).__name__}")
            return None

        # Beklenen anahtarlar mevcut mu?
        required_keys = {"xgb"}
        missing = required_keys - set(model.keys())

        if missing:
            print(f"❌ [load_model] Eksik model bileşenleri: {', '.join(missing)}")
            return None

        print("✅ Model başarıyla yüklendi.")
        return model

    except Exception as e:
        print(f"❌ [load_model] Yükleme esnasında kritik hata: {e}")
        return None
    
def validate_output(result, schema):
    for key in schema:
        if key not in result:
            result[key] = None
    return result

def determine_trend(adx, rsi, flow_signal):
    
    # =========================
    # 1. TREND GÜCÜ SINIFI
    # =========================
    if adx >= 30:
        trend_strength = "strong"
    elif adx >= 20:
        trend_strength = "weak"
    else:
        trend_strength = "none"

    # =========================
    # 2. YÖN TESPİTİ (MULTI FACTOR)
    # =========================
    score = 0

    # RSI katkısı
    if rsi > 55:
        score += 1
    elif rsi < 45:
        score -= 1

    # FLOW katkısı (KRİTİK)
    if flow_signal > 0:
        score += 1
    elif flow_signal < 0:
        score -= 1

    # =========================
    # 3. SONUÇ HARİTASI
    # =========================
    if trend_strength == "none":
        return "⚪ Yatay / Belirsiz Rejim"

    if trend_strength == "weak":
        if score >= 1:
            return "🟡 Zayıf Yükseliş Eğilimi"
        elif score <= -1:
            return "🟠 Zayıf Düşüş Eğilimi"
        else:
            return "⚪ Kararsız / Sıkışma"

    if trend_strength == "strong":
        if score >= 2:
            return "🟢 Güçlü Yükseliş Trendi"
        elif score <= -2:
            return "🔴 Güçlü Düşüş Trendi"
        elif score == 1:
            return "🟡 Trend var ama momentum zayıf"
        elif score == -1:
            return "🟠 Trend aşağı ama alım tepki ihtimali"
        else:
            return "⚪ Güçlü trend içinde konsolidasyon"


# =============================================================================
# 1. ERKEN DAĞITIM UYARISI & KÂR KORUMA HESAPLAMASI
# =============================================================================
def calculate_exit_strategy(price, rsi, buy_pressure, flow_signal, ma50, gunluk_getiri, direnc):
    """
    Ticaret Standartlarında Gelişmiş Satış ve Kâr Koruma Modülü.
    Tepe yorgunluklarını ve kurumsal dağıtımları tespit ederek erken uyarı üretir.
    """
    kar_koruma_skoru = 0
    
    if rsi > 50:
        if rsi > 70:
            kar_koruma_skoru += 30
        if buy_pressure < 0.50:
            kar_koruma_skoru += 30
        if flow_signal < 0:
            kar_koruma_skoru += 40

        if kar_koruma_skoru < 30:
            kar_koruma_durumu = "🟢 Pozisyonu Koru / Tut"
        elif kar_koruma_skoru < 60:
            kar_koruma_durumu = "🟡 Kârı Koru (Yakın Stop)"
        else:
            kar_koruma_durumu = "🟠 Kademeli Satış Düşünülmeli"

        # Zirve Yorgunluğu
        zirve_yorgunlugu_sinyali = "🟢 Normal (Zirve Baskısı Yok)"
        if (price > ma50 * 1.08) and (rsi > 65) and (flow_signal < 0):
            zirve_yorgunlugu_sinyali = "⚠️ YÜKSELİŞ DEVAM EDİYOR AMA PARA ÇIKIŞI BAŞLADI (Mal Dağıtımı!)"

        # Gün İçi Alarm
        gun_ici_alarm = "🟢 Normal"
        if gunluk_getiri > 4 and flow_signal < 0:
            gun_ici_alarm = "💰 KÂR KORUMA MODU AKTİF (Kademeli satış düşünülebilir)"

        # Tepe Skoru
        tepe_skoru = 0
        if rsi > 70:
            tepe_skoru += 25
        if flow_signal < 0:
            tepe_skoru += 25
        if buy_pressure < 0.48:
            tepe_skoru += 25
        if price >= direnc * 0.98:
            tepe_skoru += 25

        if tepe_skoru <= 25:
            tepe_bolgesi_durumu = "🟢 Normal Bölge"
        elif tepe_skoru <= 50:
            tepe_bolgesi_durumu = "🟡 Dikkat Seviyesi"
        elif tepe_skoru <= 75:
            tepe_bolgesi_durumu = "🟠 Kâr Koru / İzle"
        else:
            tepe_bolgesi_durumu = "🔴 GÜÇLÜ SATIŞ BÖLGESİ (Tepe Noktası)"
            
    else:
        kar_koruma_skoru = 0
        tepe_skoru = 0
        kar_koruma_durumu = "🟢 Pozisyonu Koru / Tut"
        zirve_yorgunlugu_sinyali = "🟢 Normal (Zirve Baskısı Yok)"
        gun_ici_alarm = "🟢 Normal"
        tepe_bolgesi_durumu = "🟢 Normal Bölge"

    # Karar Yapısı ve Yorum Üretimi
    if flow_signal < -0.15 and buy_pressure < 0.45:
        ex_action = "BEKLE/SAT"
        ex_comment = ("Tepe oluşumu yok ancak hisse üzerinde güçlü satış baskısı bulunuyor. "
                      "Alıcılar zayıf, para çıkışı devam ediyor. Yeni pozisyon için erken.")
    elif tepe_skoru >= 75 or kar_koruma_skoru >= 60:
        ex_action = "KADEMELİ SAT"
        ex_comment = (f"Kritik Uyarı: Tepe skoru (%{tepe_skoru}) and Erken Dağıtım riski çok yüksek. "
                      f"Fiyat yükseliyor görünse de akıllı para çıkışı (Flow Signal: {flow_signal:.3f}) onaylandı. "
                      f"Kar realizasyonu rasyoneldir.")
    elif tepe_skoru >= 50 or gun_ici_alarm != "🟢 Normal":
        ex_action = "KÂR KORU / İZLE"
        ex_comment = ("Hissede gün içi güçlü marj yakalandı ancak üst kademelerde "
                      "yorgunluk ve sığlaşma emareleri var. İz süren stopu yukarı çekerek "
                      "pozisyonu koruyun.")
    else:
        ex_action = "TUT"
        ex_comment = ("Hissede tepe yorgunluğu veya kurumsal mal boşaltma sinyali "
                      "bulunmuyor. Trend sağlıklı şekilde devam ediyor veya dip arayışı hakim.")

    return {
        "kar_koruma_skoru": kar_koruma_skoru,
        "kar_koruma_durumu": kar_koruma_durumu,
        "tepe_skoru": tepe_skoru,
        "tepe_bolgesi_durumu": tepe_bolgesi_durumu,
        "zirve_yorgunlugu": zirve_yorgunlugu_sinyali,
        "gun_ici_alarm": gun_ici_alarm,
        "exit_strategy_action": ex_action,
        "exit_strategy_comment": ex_comment
    }

def get_live_price(symbol: str) -> float | None:
    """Hisse senedinin güncel fiyatını almaya çalışır.

    Öncelik sırası:
    1. fast_info (en hızlı)
    2. info
    3. Son kapanış (download)

    Başarısız olursa None döner.
    """
    yf_symbol = symbol if symbol.endswith(".IS") else f"{symbol}.IS"

    try:
        ticker = yf.Ticker(yf_symbol)

        # =====================================================================
        # 1. ÖNCELİK: FAST_INFO
        # =====================================================================
        try:
            if hasattr(ticker, "fast_info"):
                fi = ticker.fast_info

                if isinstance(fi, dict):
                    live_price = fi.get("lastPrice") or fi.get("regularMarketPrice")
                else:
                    live_price = getattr(fi, "lastPrice", None) or getattr(fi, "regularMarketPrice", None)

                if live_price is not None and live_price > 0:
                    return float(live_price)
        except Exception:
            pass

        # =====================================================================
        # 2. FALLBACK: INFO
        # =====================================================================
        try:
            info = ticker.info
            live_price = (
                info.get("regularMarketPrice")
                or info.get("currentPrice")
                or info.get("previousClose")
            )

            if live_price is not None and live_price > 0:
                return float(live_price)
        except Exception:
            pass

        # =====================================================================
        # 3. SON ÇARE: DOWNLOAD
        # =====================================================================
        df = yf.download(
            yf_symbol,
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if "Close" in df.columns:
                close_price = float(df["Close"].iloc[-1])
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
    close = df["Close"]

    df["rsi"] = ta.momentum.RSIIndicator(close, 14).rsi()

    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["ma50"] = close.rolling(50).mean()
    df["ma200"] = close.rolling(200).mean()

    df["return"] = close.pct_change()
    df["volatility"] = df["return"].rolling(10).std()

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


def detect_regime(adx, rsi, flow):
    if adx > 25 and flow > 0: return "TREND"
    elif adx < 18: return "SIDEWAYS"
    elif rsi > 70 and flow < 0: return "DISTRIBUTION"
    elif rsi < 45 and flow > 0: return "ACCUMULATION"
    return "UNKNOWN"

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
        "suni_kod": 0,
        "kar_koruma_skoru": 0, "kar_koruma_durumu": "N/A", "tepe_skoru": 0,
        "tepe_bolgesi_durumu": "N/A", "zirve_yorgunlugu": "N/A", "gun_ici_alarm": "N/A",
        "exit_strategy_action": "BEKLE/SAT", "exit_strategy_comment": "Veri Yetersiz"
    }

def _prepare_data(
    df: pd.DataFrame,
    features: list | None,
    model: dict | None
) -> tuple[pd.DataFrame, list]:
    """Analiz öncesi veri hazırlama sürecini yönetir.

    - OHLCV kolonlarını sayısala çevirir
    - Teknik indikatörleri hesaplar
    - NaN / Inf temizliği yapar
    - Model ile feature uyumluluğunu kontrol eder
    - Eksik feature'ları oluşturur
    - Feature sıralamasını korur
    """
    # =====================================================================
    # 0. GİRİŞ KONTROLÜ
    # =====================================================================
    if df is None or df.empty:
        return pd.DataFrame(), []

    df = df.copy()

    try:
        # =====================================================================
        # 1. SAYISAL KOLON DÖNÜŞÜMÜ
        # =====================================================================
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # =====================================================================
        # 2. TEKNİK İNDİKATÖRLER
        # =====================================================================
        df = add_indicators(df)

        # =====================================================================
        # 3. INF / NAN TEMİZLİĞİ
        # =====================================================================
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        if df.empty:
            return pd.DataFrame(), []

        # =====================================================================
        # 4. FEATURE LİSTESİ
        # =====================================================================
        features = [] if features is None else list(features)

        # =====================================================================
        # 5. MODEL UYUMLULUK KONTROLÜ
        # =====================================================================
        if model is not None and isinstance(model, dict) and "xgb" in model:
            try:
                xgb_model = model["xgb"]

                # Model feature isimlerini tutuyorsa öncelik ver
                if hasattr(xgb_model, "feature_names") and xgb_model.feature_names:
                    features = list(xgb_model.feature_names)

                # Feature sayısını kontrol et
                elif hasattr(xgb_model, "n_features_in_"):
                    expected_n = xgb_model.n_features_in_
                    if len(features) > expected_n:
                        features = features[:expected_n]
            except Exception:
                pass

        # =====================================================================
        # 6. EKSİK FEATURE OLUŞTUR
        # =====================================================================
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0.0

        # =====================================================================
        # 7. FEATURE DOĞRULAMA
        # =====================================================================
        valid_features = [f for f in features if f in df.columns]

        if not valid_features:
            return pd.DataFrame(), []

        return df, valid_features

    except Exception:
        return pd.DataFrame(), []


def _calculate_order_flow(df):
    price_diff = df["Close"].diff().fillna(0)
    real_lots = df["Volume"].values
    diff_vals = price_diff.values

    buy_volume = float(np.where(diff_vals > 0, real_lots, 0).sum())
    sell_volume = float(np.where(diff_vals < 0, real_lots, 0).sum())

    total_volume = buy_volume + sell_volume + 1e-9

    buy_pressure = buy_volume / total_volume
    sell_pressure = sell_volume / total_volume

    flow_raw = (buy_volume - sell_volume) / total_volume
    flow_index = np.tanh(flow_raw)
    flow_signal = flow_raw
    
    df_lots = df["Volume"] / (df["Close"] + 1e-9)
    avg_lots_20 = df_lots.rolling(window=20).mean().iloc[-1]
    last_lot = df_lots.iloc[-1]
    last_price_change = price_diff.iloc[-1]
    
    kalicilik_durumu = "Kalıcı / Dengeli"
    suni_kod = 0  
    
    if last_price_change > 0 and last_lot < (avg_lots_20 * 0.8):
        kalicilik_durumu = "⚠️ SUNİ YÜKSELİŞ (Düşük Hacim Tuzak)"
        suni_kod = 1
    elif last_price_change < 0 and last_lot < (avg_lots_20 * 0.8):
        kalicilik_durumu = "⚠️ SUNİ DÜŞÜŞ (Hacimsiz Silkeleme)"
        suni_kod = -1
    elif last_price_change > 0 and last_lot > (avg_lots_20 * 1.3):
        kalicilik_durumu = "🚀 KALICI YÜKSELİŞ (Hacim Onaylı)"
        suni_kod = 2
        
    return {
        "buy_volume": buy_volume, "sell_volume": sell_volume,
        "buy_pressure": buy_pressure, "sell_pressure": sell_pressure,
        "flow_raw": flow_raw, "flow_index": flow_index, "flow_signal": flow_signal,
        "kalicilik_durumu": kalicilik_durumu, "suni_kod": suni_kod
    }

def _calculate_smart_money(last, close, flow_signal):
    rsi = float(last["rsi"])
    macd = float(last["macd"])
    macd_signal = float(last["macd_signal"])
    ma200 = float(last["ma200"])
    
    smart = 0
    if macd > macd_signal: smart += 20
    if close > ma200: smart += 20
    if rsi < 50: smart += 10
    if rsi > 70: smart -= 10
    if flow_signal > 0: smart += 15

    return float(np.clip(smart, 0, 100))

def _determine_regime(adx, rsi, flow_signal):
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
    """Ana analiz motoru.
    
    Veriyi indirir, temizler, ön işlemeden geçirir ve XGBoost modeliyle
    tahmin üreterek sonuçları bir sözlük olarak döndürür.
    """
    market_open = is_market_open(strict=True)

    # =====================================================================
    # 1. VERİ İNDİRME
    # =====================================================================
    df_raw = yf.download(
        f"{symbol}.IS",
        period="60d",
        interval="1d",
        progress=False,
        auto_adjust=True
    )

    if df_raw is None or df_raw.empty:
        return None

    # =====================================================================
    # 2. PIPELINE (TEMİZLİK VE ÖN İŞLEME)
    # =====================================================================
    df_clean = _clean_yfinance_df(df_raw)
    df_prepared, valid_features = _prepare_data(df_clean, features, model)

    if df_prepared.empty or not valid_features:
        return None

    # =====================================================================
    # 3. MODEL KONTROLÜ
    # =====================================================================
    if "xgb" not in model or not hasattr(model["xgb"], "predict"):
        return None

    # =====================================================================
    # 4. LAST ROW (GÜVENLİ HİZALAMA VE REINDEX)
    # =====================================================================
    try:
        last_row = df_prepared.reindex(columns=valid_features).iloc[[-1]].copy()
        last_row = last_row.fillna(0)
    except Exception:
        return None

    # =====================================================================
    # 5. PREDICTION (TAHMİN VE TİP DÖNÜŞÜMÜ)
    # =====================================================================
    prediction = model["xgb"].predict(last_row)
    pred_value = float(prediction[0]) if hasattr(prediction, "__len__") else float(prediction)

    return {
        "symbol": symbol,
        "prediction": pred_value,
        "market_open": market_open
    }

def no_trade_response():
    return {"signal": "NO_TRADE", "note": "Market closed"}


# Sadece ilk satıra , regime_text parametresi eklendi (Hata Çözüldü)
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
    atr = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"]
    ).average_true_range().iloc[-1]

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
        rr = reward / risk

    rr = np.clip(rr, 0.0, 5.0)
    return destek, direnc, rr, risk, reward, piyasa_notu


def _get_edge_case_comment(rsi, flow_index):
    """RSI ve Para Akışı uç durumlarına göre kritik uyarı mesajı üretir."""
    edge_comment = "🟢 Normal piyasa koşulu. "
    if rsi < 25 and flow_index < 0:
        edge_comment = "⚠️ Aşırı satım + para çıkışı devam ediyor. Panik satış / trend devam riski var. "
    elif rsi < 25 and flow_index > 0:
        edge_comment = "🟢 Aşırı satım + para girişi var. Rebound ihtimali artıyor. "
    elif rsi > 70 and flow_index < 0:
        edge_comment = "⚠️ Aşırı alım + dağıtım riski oluşuyor. "
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

    # Sinyal Seçimi
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


def _generate_ai_comment(signal, edge_comment, trend_text, flow, dynamic_stop, dynamic_tp, destek, direnc):
    """Oluşan sinyallere ve piyasa rejimine göre nihai yapay zeka yorumunu oluşturur."""
    if signal in ["🚀 ÇOK GÜÇLÜ AL", "🟢 AL"]:
        return (
            f"{edge_comment}"
            f"Hisse net bir {trend_text} içerisinde ve güçlü teknik yapı sergiliyor. "
            f"Alıcı baskısı (%{flow['buy_pressure']*100:.1f}) devam ediyor. "
            f"Stop-loss: {dynamic_stop:.2f}, hedef: {dynamic_tp:.2f}."
        )
    elif signal == "🟡 İZLE (POTANSİYEL AL)":
        return (
            f"{edge_comment}"
            f"Hisse {trend_text} içerisinde. Alıcı baskısı %{flow['buy_pressure']*100:.1f}. "
            f"Direnç seviyesi {dynamic_tp:.2f} yakın, dikkatli takip gerekli."
        )
    elif signal == "🟡 İZLE":
        return (
            f"{edge_comment}"
            f"Piyasa {trend_text}. Sıkışma devam ediyor. "
            f"Destek {destek:.2f} / direnç {direnc:.2f} izlenmeli."
        )
    elif signal == "⚪ NÖTR":
        return (
            f"{edge_comment}"
            f"Yönsüz piyasa ({trend_text}). Net kırılım beklenmeli."
        )
    else:
        return (
            f"{edge_comment}"
            f"Satış baskısı güçlü (%{flow['sell_pressure']*100:.1f}). "
            f"Risk yönetimi ön planda."
        )
    
    

def analyze(df, model, features, symbol="THYAO"):
    """
    Gelişmiş yapay zeka ve sipariş akışı (order flow) entegrasyonlu 
    hisse senedi analiz motoru. (Production-Safe & Live Consistent)
    """
    
    # -------------------------------------------------------------------------
    # 1. DATA GUARD (Öncelikli Veri Kontrolü)
    # -------------------------------------------------------------------------
    if df is None or len(df) < 30:
        return _get_empty_result()

    df, features = _prepare_data(df, features, model)
    if df.empty:
        return _get_empty_result()

    # -------------------------------------------------------------------------
    # 2. LIVE PRICE & SEANS SONU GÜVENLİK YAMASI (Her Zaman Çalışmalı)
    # -------------------------------------------------------------------------
    live_price = None
    try:
        # Seans açıksa anlık fiyatı çekmeyi dene
        live_price = get_live_price(symbol)
    except Exception:
        live_price = None

    # 💡 KRİTİK DÜZELTME: Seans kapalıysa, gün içi 5m barlarındaki yfinance kaymasını 
    # engellemek için günlük (1d) resmi kapanışı çekip canlı fiyat olarak belirliyoruz.
    if not is_market_open():
        try:
            yf_symbol = f"{symbol}.IS" if not symbol.endswith(".IS") else symbol
            df_daily = yf.download(yf_symbol, period="1d", interval="1d", auto_adjust=True, progress=False)
            if not df_daily.empty:
                resmi_kapanis = float(df_daily["Close"].iloc[-1])
                if resmi_kapanis > 0:
                    live_price = resmi_kapanis
        except Exception:
            pass

    # DataFrame'in son barını gerçek fiyata göre manipüle ediyoruz
    if live_price is not None and live_price > 0:
        close = float(live_price)
        idx = df.index[-1]
        df.loc[idx, "Close"] = close
        
        # High/Low sınırlarını yeni fiyata göre koruyoruz
        if close > df.loc[idx, "High"]: df.loc[idx, "High"] = close
        if close < df.loc[idx, "Low"]:  df.loc[idx, "Low"] = close
    else:
        close = float(df.iloc[-1]["Close"])

    base_last = df.iloc[-1].copy()

    # -------------------------------------------------------------------------
    # 3. MARKET GUARD (Arayüz Esnekliği İçin Konumu Değiştirildi)
    # -------------------------------------------------------------------------
    if not is_market_open():
        # Gece/hafta sonu arayüzün kilitlenmemesi ve son analizi basması için pass geçiyoruz.
        pass

    # -------------------------------------------------------------------------
    # 4. MODEL INPUT (Yeni Fiyata Göre Tam Tutarlı)
    # -------------------------------------------------------------------------
    X = df[features].iloc[-1:].values

    # Hibrit Model Olasılık Hesaplaması (XGBoost + LightGBM Ensemble)
    try:
        proba_xgb = model["xgb"].predict_proba(X)[0][1]
        proba_lgbm = model["lgbm"].predict_proba(X)[0][1]
        proba = (proba_xgb + proba_lgbm) / 2
    except Exception:
        proba = 0.5  # Güvenli fallback

    # -------------------------------------------------------------------------
    # 5. INDICATORS (Güncellenen Fiyat Üzerinden Yeniden Hesaplama)
    # -------------------------------------------------------------------------
    adx_series = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
    adx = float(adx_series.iloc[-1]) if len(adx_series) and not pd.isna(adx_series.iloc[-1]) else 25.0

    rsi_series = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    rsi = float(rsi_series.iloc[-1]) if len(rsi_series) and not pd.isna(rsi_series.iloc[-1]) else 50.0

    # -------------------------------------------------------------------------
    # 6. ORDER FLOW & SCORE GENERATION
    # -------------------------------------------------------------------------
    flow = _calculate_order_flow(df)
    smart_money = float(_calculate_smart_money(base_last, close, flow["flow_signal"]))

    # Güven Skoru: %60 Yapay Zeka Olasılığı + %40 Akıllı Para Hacim Dengesi
    guven_skoru = float(np.clip(proba * 100 * 0.60 + smart_money * 0.40, 0, 100))

    # -------------------------------------------------------------------------
    # 7. REGIME + TREND
    # -------------------------------------------------------------------------
    regime_text = _determine_regime(adx, rsi, flow["flow_signal"])
    regime, trend_text = _get_market_regime_and_trend(
        adx, rsi, flow["flow_signal"], regime_text
    )

    # -------------------------------------------------------------------------
    # 8. KALICILIK DURUMU (Karakteristik Analiz)
    # -------------------------------------------------------------------------
    flow_index = float(flow["flow_index"])

    if adx < 20 and flow_index < -0.1:
        kalicilik_durumu = "⚪ Zayıf / Yatay"
    elif adx > 25 and flow_index > 0.2:
        kalicilik_durumu = "🚀 KALICI YÜKSELİŞ"
    elif adx > 25 and flow_index < -0.2:
        kalicilik_durumu = "🔴 DAĞILIM / ZAYIFLAMA"
    elif adx < 20 and abs(flow_index) < 0.1:
        kalicilik_durumu = "⚪ SIKIŞMA / KARARSIZ"
    else:
        kalicilik_durumu = "⚪ DENGELİ"

    # -------------------------------------------------------------------------
    # 9. RISK ENGINE
    # -------------------------------------------------------------------------
    veri_donuk = bool(df["Close"].tail(20).nunique() <= 1)
    destek, direnc, rr, risk, reward, piyasa_notu = _calculate_risk_metrics(
        df, close, veri_donuk
    )

    # -------------------------------------------------------------------------
    # 10. EDGE CASES
    # -------------------------------------------------------------------------
    edge_comment = _get_edge_case_comment(rsi, flow_index)
    if 45 <= rsi <= 55 and abs(flow_index) < 0.05:
        edge_comment = "⚪ Düşük volatilite / sıkışma bölgesi."

    # -------------------------------------------------------------------------
    # 11. SCORE & SIGNAL ENGINE
    # -------------------------------------------------------------------------
    score, signal, color = _calculate_scores_and_signal(
        proba, smart_money, rsi, flow_index, regime, regime_text, flow["buy_pressure"]
    )

    # -------------------------------------------------------------------------
    # 12. METRICS GENERATION (Karne Yapısı)
    # -------------------------------------------------------------------------
    stock_score = float(np.clip(score * 40 + (smart_money / 100) * 30 + min(rr / 3, 1) * 30, 0, 100))
    final_score = float(np.clip(score * 60 + (smart_money / 100) * 25 + (rr / 5) * 15, 0, 100))

    grade = (
        "A+" if final_score >= 85 else
        "A" if final_score >= 75 else
        "B" if final_score >= 65 else "C"
    )

    # -------------------------------------------------------------------------
    # 13. DYNAMIC STOP / TAKE PROFIT
    # -------------------------------------------------------------------------
    volatility_ratio = risk / close
    dynamic_stop = float(close * 0.95 if volatility_ratio < 0.02 else destek)
    dynamic_tp = float(close * 1.05 if (reward / close < 0.02) else direnc)

    # -------------------------------------------------------------------------
    # 14. EXIT STRATEGY (Erken Çıkış Sistemi Parametreleri)
    # -------------------------------------------------------------------------
    gunluk_getiri = 0.0
    if len(df) >= 2:
        prev = float(df["Close"].iloc[-2])
        if prev > 0:
            gunluk_getiri = float(((close - prev) / prev) * 100)

    ma50_val = (
        float(base_last["ma50"])
        if pd.notna(base_last.get("ma50"))
        else close
    )

    exit_data = calculate_exit_strategy(
        price=close,
        rsi=rsi,
        buy_pressure=float(flow["buy_pressure"]),
        flow_signal=float(flow["flow_signal"]),
        ma50=ma50_val,
        gunluk_getiri=gunluk_getiri,
        direnc=float(direnc)
    )

    # -------------------------------------------------------------------------
    # 15. AI STRATEGY COMMENT GENERATION
    # -------------------------------------------------------------------------
    ai_comment = _generate_ai_comment(
        signal, edge_comment, trend_text, flow, dynamic_stop, dynamic_tp, destek, direnc
    )

    # -------------------------------------------------------------------------
    # 16. OUTPUT DIRECTORY (Strict Type Casting & Streamlit Safe)
    # -------------------------------------------------------------------------
    return {
        # Gerçek Resmi Fiyatlar ve İndikatörler
        "price": float(close),
        "rsi": float(rsi),
        "adx": float(adx),

        # Piyasa Eğilimleri
        "regime": str(regime_text),
        "trend_text": str(trend_text),
        "kalicilik_durumu": str(kalicilik_durumu),
        "piyasa_notu": str(piyasa_notu),

        # Emir Akışı Detayları
        "flow": float(flow_index),
        "flow_index": float(flow_index),
        "flow_raw": float(flow["flow_raw"]),
        "flow_signal": float(flow["flow_signal"]),
        "buy_volume": int(flow["buy_volume"]),
        "sell_volume": int(flow["sell_volume"]),
        "buy_pressure": float(flow["buy_pressure"]),
        "sell_pressure": float(flow["sell_pressure"]),
        "smart_money": float(smart_money),
        "suni_kod": int(flow.get("suni_kod", 0)),
        
        # Algoritmik Skorlar ve Kararlar
        "guven_skoru": float(guven_skoru),
        "rr": float(rr),
        "score": float(score),
        "final_score": float(final_score),
        "grade": str(grade),
        "signal": str(signal),
        "color": str(color),

        # Risk Yönetimi Hatları
        "destek": float(destek),
        "direnc": float(direnc),
        "stop_loss": float(dynamic_stop),
        "take_profit": float(dynamic_tp),

        # Hisse Puanlama ve Eylem Planı
        "stock_score": float(stock_score),
        "hisse_karnesi": float(stock_score),
        "strategy_action": str(
            "YAKIN İZLE" if "İZLE" in signal
            else ("AL" if "AL" in signal else "BEKLE/SAT")
        ),

        # Yorum Katmanları
        "ai_comment": str(ai_comment),
        "edge_comment": str(edge_comment),

        # Tepe ve Erken Çıkış Algoritması Çıktıları
        "kar_koruma_skoru": float(exit_data.get("kar_koruma_skoru", 0)),
        "kar_koruma_durumu": str(exit_data.get("kar_koruma_durumu", "🟢 Pozisyonu Koru / Tut")),
        "tepe_skoru": float(exit_data.get("tepe_skoru", 0)),
        "tepe_bolgesi_durumu": str(exit_data.get("tepe_bolgesi_durumu", "🟢 Normal Bölge")),
        "zirve_yorgunlugu": str(exit_data.get("zirve_yorgunlugu", "🟢 Normal")),
        "gun_ici_alarm": str(exit_data.get("gun_ici_alarm", "🟢 Normal")),
        "exit_strategy_action": str(exit_data.get("exit_strategy_action", "BEKLE/SAT")),
        "exit_strategy_comment": str(exit_data.get("exit_strategy_comment", ""))
    }
######=========================================================================
#   Grafik Ekranı
#####=========================================================================
def create_technical_chart(df, result):

    # Orijinal df'i korumak ve kolon hatalarını önlemek için kopyalıyoruz
    df_chart = add_indicators(df.copy())
    
    # MACD Histogram hesaplamasını güvenli bir şekilde df_chart üzerinde yapıyoruz
    df_chart["macd_hist"] = df_chart["macd"] - df_chart["macd_signal"]

    # Matris yapısını kuruyoruz
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03, # Paneller arası boşluk hafif artırıldı (daha temiz görünüm)
        row_heights=[0.50, 0.15, 0.15, 0.20],
        subplot_titles=("Fiyat Grafiği", "Hacim", "RSI (Göreceli Güç Endeksi)", "MACD")
    )

    # =========================
    # CANDLESTICK (Mum Grafiği)
    # =========================
    fig.add_trace(
        go.Candlestick(
            x=df_chart.index,
            open=df_chart["Open"],
            high=df_chart["High"],
            low=df_chart["Low"],
            close=df_chart["Close"],
            name="Fiyat",
            increasing_line_color='#26a69a', # Profesyonel TradingView yeşili
            decreasing_line_color='#ef5350'  # Profesyonel TradingView kırmızısı
        ),
        row=1, col=1
    )

    # =========================
    # MA50 (Hareketli Ortalama)
    # =========================
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=df_chart["ma50"],
            mode="lines",
            name="MA50",
            line=dict(color='#ff9800', width=1.5) # Turuncu tonu
        ),
        row=1, col=1
    )
    
    # =========================
    # MA200 (Hareketli Ortalama)
    # =========================
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=df_chart["ma200"],
            mode="lines",
            name="MA200",
            line=dict(color='#2196f3', width=2) # Mavi tonu
        ),
        row=1, col=1
    )
    
    # =========================
    # ALIM / SATIM OKLARI (Kesişim Mantığı ile Profesyonelleştirildi)
    # =========================
    # Sadece ilk kesişim (crossover/crossunder) anlarında sinyal üretmek için .shift() eklendi.
    # Böylece her gün ok basmak yerine sadece sinyalin ilk geldiği güne tek bir ok basılır.
    macd_cross_up = (df_chart["macd"] > df_chart["macd_signal"]) & (df_chart["macd"].shift(1) <= df_chart["macd_signal"].shift(1))
    macd_cross_down = (df_chart["macd"] < df_chart["macd_signal"]) & (df_chart["macd"].shift(1) >= df_chart["macd_signal"].shift(1))

    buy_signal = macd_cross_up & (df_chart["rsi"] < 70)
    sell_signal = macd_cross_down & (df_chart["rsi"] > 30)

    # Alım işaretleri (Yeşil Yukarı Ok)
    fig.add_trace(
        go.Scatter(
            x=df_chart.index[buy_signal],
            y=df_chart["Low"][buy_signal] * 0.98, # Okların mumun altında düzgün görünmesi için %2 ofset
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=14,
                color="#00e676",
                line=dict(color="#003300", width=1)
            ),
            name="AL Sinyali"
        ),
        row=1, col=1
    )
    
    # Satım işaretleri (Kırmızı Aşağı Ok)
    fig.add_trace(
        go.Scatter(
            x=df_chart.index[sell_signal],
            y=df_chart["High"][sell_signal] * 1.02, # Okların mumun üstünde düzgün görünmesi için %2 ofset
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                size=14,
                color="#ff1744",
                line=dict(color="#330000", width=1)
            ),
            name="SAT Sinyali"
        ),
        row=1, col=1
    )

    # =========================
    # HACİM (Volume)
    # =========================
    # Profesyonel görünüm için mum rengine göre hacim barlarını renklendiriyoruz
    volume_colors = ['#26a69a' if c >= o else '#ef5350' for o, c in zip(df_chart["Open"], df_chart["Close"])]
    
    fig.add_trace(
        go.Bar(
            x=df_chart.index,
            y=df_chart["Volume"],
            name="Hacim",
            marker_color=volume_colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # =========================
    # RSI PANELİ
    # =========================
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=df_chart["rsi"],
            mode="lines",
            name="RSI",
            line=dict(color='#9c27b0', width=1.5) # Mor RSI çizgisi
        ),
        row=3, col=1
    )

    # RSI Sınır Çizgileri
    fig.add_hline(y=70, row=3, col=1, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig.add_hline(y=30, row=3, col=1, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig.update_yaxes(range=[10, 90], row=3, col=1) # Eksen sınırları optimize edildi
    
    # =========================
    # MACD PANELİ & HİSTOGRAM
    # =========================
    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=df_chart["macd"],
            name="MACD",
            line=dict(color='#2196f3', width=1.5)
        ),
        row=4, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=df_chart["macd_signal"],
            name="Sinyal",
            line=dict(color='#ff9800', width=1.5)
        ),
        row=4, col=1
    )
    
    # Histogram Barları (Pozitif/Negatif renk ayrımı ile)
    hist_colors = ['#26a69a' if val >= 0 else '#ef5350' for val in df_chart["macd_hist"]]
    fig.add_trace(
        go.Bar(
            x=df_chart.index,
            y=df_chart["macd_hist"],
            name="Histogram",
            marker_color=hist_colors,
            opacity=0.6
        ),
        row=4, col=1
    )

    # =========================
    # FİNANSAL SEVİYELER (Destek, Direnç, SL, TP)
    # =========================
    line_styles = {
        "destek": dict(color="#26a69a", dash="dot"),
        "direnc": dict(color="#ef5350", dash="dot"),
        "stop_loss": dict(color="#ff1744", dash="dash"),
        "take_profit": dict(color="#00e676", dash="dash")
    }
    
    labels = {"destek": "Destek", "direnc": "Direnç", "stop_loss": "Stop Loss (SL)", "take_profit": "Take Profit (TP)"}

    for key, style in line_styles.items():
        if key in result and result[key] is not None:
            fig.add_hline(
                y=result[key],
                row=1, col=1,
                line_dash=style["dash"],
                line_color=style["color"],
                annotation_text=f"{labels[key]}: {result[key]:.2f}",
                annotation_position="top left",
                annotation_font=dict(color=style["color"], size=10)
            )

    # =========================
    # SON FİYAT ETİKETİ
    # =========================
    son_fiyat = float(df_chart["Close"].iloc[-1])
    fig.add_annotation(
        x=df_chart.index[-1],
        y=son_fiyat,
        text=f"Son: {son_fiyat:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#ffffff",
        ax=-50, ay=-30,
        font=dict(color="#ffffff", size=11),
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor="#ffffff",
        borderwidth=1,
        row=1, col=1
    )

    # =========================
    # GÖRÜNÜM VE LAYOUT (Birleştirildi, Çelişkiler Giderildi)
    # =========================
    fig.update_yaxes(title_text="Fiyat ($)", row=1, col=1)
    fig.update_yaxes(title_text="Hacim", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=1000, # İdeal ve dengeli bir yükseklik ayarlandı
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        margin=dict(l=50, r=50, t=70, b=50) # Grafik kenar boşlukları optimize edildi
    )

    return fig

    



# ==============================================================================
# PYQT5 MASAÜSTÜ PENCERE SINIFI (SENİN VERDİĞİN OTOMATİK BAŞLATICI)
# ==============================================================================
if PYQT_AVAILABLE:
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
# 4. STREAMLIT MASAÜSTÜ UYGULAMA PANELİ (MANTIK VE CSS KORUNDU)
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
        # Otomatik yenileme motoru
        st_autorefresh(interval=60000, key="refresh_sekme2")

        # =====================================================================
        # 0. SESSION STATE (HAFIZA) BAŞLANGIÇ TANIMLARI
        # =====================================================================
        if "aktif_hisse" not in st.session_state:
            st.session_state.aktif_hisse = ""

        # =====================================================================
        # 1. AKILLI ARAMA FORMU (Kutuyu Enter'dan Sonra Temizleyen Yapı)
        # =====================================================================
        with st.form(key="hisse_arama_formu", clear_on_submit=True):
            hisse_input = st.text_input("Hisse Senedi Kodu (Örn: THYAO, TUREX):")
            submit_button = st.form_submit_button("🔍 Analiz Et", use_container_width=False)
            
            if submit_button and hisse_input:
                # Kutudan gelen veriyi temizleyip hafızaya alıyoruz
                st.session_state.aktif_hisse = hisse_input.strip().upper()

        # =====================================================================
        # 2. ANA ÇALIŞMA BLOKU (Hafızada aktif bir hisse varsa çalışır)
        # =====================================================================
        if st.session_state.aktif_hisse:
            hisse_kodu = st.session_state.aktif_hisse
            
            # Kullanıcının manuel .IS yazma ihtimaline karşı akıllı temizlik (Örn: TUREX.IS.IS faciasını önler)
            if hisse_kodu.endswith(".IS"):
                hisse_temiz = hisse_kodu.replace(".IS", "")
            else:
                hisse_temiz = hisse_kodu
                
            sembol = f"{hisse_temiz}.IS"
            
            # Kullanıcıya şu an hangi hissenin analizini gördüğünü hatırlatan etiket
            st.caption(f"🔎 Şu an incelenen hisse: **{sembol}**")

            # Piyasa durumuna göre esnek ve hatasız veri çekme stratejisi
            market_active = is_market_open(strict=True)

            if market_active:
                # Gündüz seansı: Son 5 günün 5 dakikalık barları
                p_time, i_time = "5d", "5m"
            else:
                # Akşam seansı: İndikatörlerin ve ML modelinin çökmemesi için en az 1 yıllık günlük veri
                p_time, i_time = "1y", "1d"
                
            # 1. Deneme: Standart .IS uzantısı ile veri çekme
            df_st = get_live_data(sembol, period=p_time, interval=i_time)

            # 2. Deneme (Bariyer): Eğer yfinance .IS ile boş döndüyse, uzantısız dene
            if df_st is None or df_st.empty:
                df_st = get_live_data(hisse_temiz, period=p_time, interval=i_time)
                if df_st is not None and not df_st.empty:
                    sembol = hisse_temiz

            model_st = load_model()

            # KONTROL BARIYERI: Veri seti veya model boşsa arayüzü patlatma
            if df_st is None or df_st.empty:
                st.error(f"❌ {sembol} için veri çekilemedi. Lütfen kodu veya internet bağlantınızı kontrol edin.")
            elif model_st is None:
                st.error("❌ Yapay zeka modeli sistemden yüklenemedi.")
            else:
                # Modeli ve veriyi ana analiz motoruna gönderiyoruz
                result = analyze(df_st, model_st, MODEL_FEATURES, symbol=hisse_temiz)
                
                if not result:
                    st.warning("⚠️ Analiz motoru bu hisse için sonuç üretemedi.")
                else:
                    # =============================================================
                    # ⏱️ CANLI GÜNCELLENME SAATİ
                    # =============================================================
                    su_an = datetime.now().strftime("%H:%M:%S")
                    st.markdown(f"""
                    <div style="
                        background-color: #1E1E1E; padding: 10px; border-radius: 8px; 
                        border-left: 5px solid #2196f3; margin-bottom: 20px;">
                        ⏱️ <b>Son Güncellenme Zamanı:</b> <span style="color: #2196f3; font-weight: bold;">{su_an}</span> 
                        <span style="font-size: 12px; color: #888888; margin-left: 10px;">(Kutu temizlendi, arka plan takibi aktif)</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # =============================================================
                    # 🚦 HAREKET KALICILIK & TUZAK DEDEKTÖRÜ
                    # =============================================================
                    suni_kod = result.get("suni_kod", 0)
                    status_color = "#4CAF50" if suni_kod == 2 else ("#FF9800" if suni_kod in [1, -1] else "#90A4AE")
                    status_bg = "rgba(76, 175, 80, 0.1)" if suni_kod == 2 else ("rgba(255, 152, 0, 0.1)" if suni_kod in [1, -1] else "rgba(144, 164, 174, 0.1)")

                    st.markdown(f"""
                    <div style="
                        background-color: {status_bg}; padding: 15px; border-radius: 12px; 
                        border: 1px solid {status_color}; border-left: 6px solid {status_color}; margin-bottom: 25px;">
                        <span style="font-size: 14px; color: #888888; text-transform: uppercase; font-weight: bold; letter-spacing: 1px;">Hacim ve Trend Doğrulama</span>
                        <div style="font-size: 20px; font-weight: 700; color: white; margin-top: 5px;">
                            {result.get('kalicilik_durumu', 'Veri Yok')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # =============================================================
                    # 🎯 KÂR KORUMA & ERKEN SATIŞ DİSİPLİNİ (Tepe Dedektörü)
                    # =============================================================
                    tepe_skoru = result.get("tepe_skoru", 0)
                    if tepe_skoru <= 25:
                        exit_color = "#4CAF50"
                    elif tepe_skoru <= 50:
                        exit_color = "#FFEB3B"
                    elif tepe_skoru <= 75:
                        exit_color = "#FF9800"
                    else:
                        exit_color = "#F44336"
                        
                    tavsiye_aksiyon = result.get('exit_strategy_action', result.get('exit_action', 'TUT'))

                    st.markdown(f"""
                    <div style="
                        background-color: #1A1A1A; padding: 18px; border-radius: 12px; 
                        border: 1px solid #2D2D2D; border-top: 4px solid {exit_color}; margin-bottom: 25px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 14px; color: #AAAAAA; font-weight: bold;">🎯 ERKEN ÇIKIŞ & TEPE ANALİZİ</span>
                            <span style="background-color: {exit_color}; color: #000000; padding: 2px 8px; border-radius: 20px; font-size: 12px; font-weight: 800;">
                                TEPE SKORU: %{tepe_skoru}
                            </span>
                        </div>
                        <div style="margin-top: 12px;">
                            <p style="margin: 3px 0; font-size: 14px; color: white;"><b>Tepe Bölgesi Durumu:</b> {result.get('tepe_bolgesi_durumu', 'Normal')}</p>
                            <p style="margin: 3px 0; font-size: 14px; color: white;"><b>Kâr Koruma Safhası:</b> {result.get('kar_koruma_durumu', 'Tut')}</p>
                            <p style="margin: 3px 0; font-size: 14px; color: white;"><b>Zirve Yorgunluğu:</b> {result.get('zirve_yorgunlugu', 'Normal')}</p>
                            <p style="margin: 3px 0; font-size: 14px; color: white;"><b>Gün İçi Ekstra Alarm:</b> {result.get('gun_ici_alarm', 'Normal')}</p>
                        </div>
                        <div style="margin-top: 12px; padding-top: 8px; border-top: 1px dashed #333; font-size: 15px; font-weight: bold; color: {exit_color};">
                            👉 Çıkış Aksiyon Tavsiyesi: {tavsiye_aksiyon}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # =============================================================
                    # 📊 KPI PANELİ
                    # =============================================================
                    st.markdown("## 📊 KPI Paneli")

                    def safe_float(val): return float(val) if val is not None else 0.0

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Fiyat", f"{safe_float(result.get('price')):.2f}")
                    c2.metric("RSI", f"{safe_float(result.get('rsi')):.1f}")
                    c3.metric("ADX", f"{safe_float(result.get('adx')):.1f}")
                    
                    c4, c5, c6 = st.columns(3)
                    c4.metric("Flow", f"%{safe_float(result.get('flow'))*100:.2f}")
                    c5.metric("Smart Money", f"{safe_float(result.get('smart_money')):.0f}/100")
                    c6.metric("Güven", f"%{safe_float(result.get('guven_skoru')):.0f}")

                    c7, c8, c9 = st.columns(3)
                    c7.metric("R/R", f"{safe_float(result.get('rr')):.2f}")
                    c8.metric("Skor", f"{safe_float(result.get('score')):.3f}")
                    c9.metric("Grade", result.get("grade", "N/A"), help="Hisse performans derecesi")
                    
                    # =============================================================================
                    # 2. ARZ / TALEP ANALİZİ (Boşluksuz f-string ve net TL ifadeleri yerleştirildi)
                    # =============================================================================
                    st.markdown("## 📊 Arz / Talep Analizi")
                    c10, c11, c12 = st.columns(3)
                    c10.metric("Alım Hacmi (TL)", f"{safe_float(result.get('buy_volume')):,.0f} ₺")
                    c11.metric("Satım Hacmi (TL)", f"{safe_float(result.get('sell_volume')):,.0f} ₺")
                    c12.metric("Flow Index", f"{safe_float(result.get('flow')):.3f}")

                    st.markdown("---")
                    st.markdown(f"""
                    🟢 **Alıcı Baskısı:** %{safe_float(result.get('buy_pressure'))*100:.1f}  
                    🔴 **Satıcı Baskısı:** %{safe_float(result.get('sell_pressure'))*100:.1f}
                    """)

                    # =============================================================
                    # 🚦 SIGNAL BOX
                    # =============================================================
                    st.markdown("## 🚦 Sinyal")
                    st.markdown(f"""
                    <div style="
                        background:{result.get('color', '#333')}; padding:18px; border-radius:12px;
                        text-align:center; font-size:30px; font-weight:800; color:white;">
                        {result.get('signal', 'SİNYAL YOK')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # =============================================================
                    # 🤖 AI RAPORU VE PİYASA REJİMİ
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
                    # 📈 GRAFİK VE ÖZET
                    # =============================================================
                    st.markdown("## 📈 Teknik Grafik")
                    fig = create_technical_chart(df_st, result)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("## 📋 Teknik Özet")
                    st.write(f"• Trend Durumu: {result.get('trend_text', 'Bilinmiyor')}")
                    st.write(f"• Stop Loss: {safe_float(result.get('stop_loss')):.2f}")
                    st.write(f"• Kar Al: {safe_float(result.get('take_profit')):.2f}")
                    st.write(f"• Hisse Karnesi: {safe_float(result.get('stock_score')):.0f}/100")
                    st.write(f"• Risk/Getiri Oranı: {safe_float(result.get('rr')):.2f}")
                    
                    if not df_st.empty:
                        st.write("DATA AGE (SON BAR):", df_st.index[-1])
                        st.write("CURRENT ROW COUNT:", len(df_st))
                        st.write("LAST CLOSE:", df_st["Close"].iloc[-1])

                    # =============================================================
                    # 🛠️ DEBUG PANELİ
                    # =============================================================
                    if st.checkbox("Debug"):
                        st.json(result)

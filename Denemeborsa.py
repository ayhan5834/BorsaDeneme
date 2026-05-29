# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:42:45 2026

@author: EmirAysu
"""


import os
import sys
import logging
import sqlite3

# =========================================================
# QT / PYINSTALLER AYARI
# =========================================================
if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
    qt_plugin_path = os.path.join(base_dir, "PyQt5", "Qt5", "plugins")
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_plugin_path

# =========================================================
# MATPLOTLIB AYARI
# =========================================================
import matplotlib
matplotlib.use('Agg')

logging.getLogger('matplotlib').setLevel(logging.ERROR)

# =========================================================
# KÜTÜPHANELER
# =========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import ta
import streamlit as st

from sklearn.linear_model import HuberRegressor

# =========================================================
# STREAMLIT AKTİF
# =========================================================
IS_STREAMLIT = True

# =========================================================
# VERİTABANI
# =========================================================
class Veritabani:

    def __init__(self):

        self.baglanti = sqlite3.connect(
            "takip_listesi.db",
            check_same_thread=False,
            timeout=10
        )

        self.cursor = self.baglanti.cursor()

        self.tablo_olustur()

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

            self.cursor.execute(
                """
                INSERT INTO watchlist
                (hisse_kodu, maliyet, adet)
                VALUES (?, ?, ?)
                """,
                (kod, maliyet, adet)
            )

            self.baglanti.commit()

        except sqlite3.IntegrityError:

            self.cursor.execute(
                """
                UPDATE watchlist
                SET maliyet = ?, adet = ?
                WHERE hisse_kodu = ?
                """,
                (maliyet, adet, kod)
            )

            self.baglanti.commit()

    def hisse_sil(self, kod):

        self.cursor.execute(
            "DELETE FROM watchlist WHERE hisse_kodu = ?",
            (kod,)
        )

        self.baglanti.commit()

    def listeyi_getir(self):

        self.cursor.execute(
            "SELECT hisse_kodu, maliyet, adet FROM watchlist"
        )

        return self.cursor.fetchall()

# =========================================================
# BIST LİSTESİ
# =========================================================
@st.cache_data(ttl=3600)
def dinamik_bist_listesi_yukle():

    csv_yolu = "bist_hisseler.csv"

    if os.path.exists(csv_yolu):

        df = pd.read_csv(csv_yolu)

        return df["kod"].tolist()

    return [
        "THYAO",
        "ASELS",
        "KRDMD",
        "TUPRS",
        "EREGL",
        "AKBNK",
        "SISE",
        "BIMAS",
        "SASA",
        "HEKTS"
    ]

TUM_BIST = dinamik_bist_listesi_yukle()

# =========================================================
# TAHMİN MOTORU
# =========================================================
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

        gelecek_gunler = pd.DataFrame({
            'gun': range(son_gun_index + 1, son_gun_index + 6)
        })

        tahmin_serisi = model.predict(gelecek_gunler)

        return tahmin_serisi[-1], tahmin_serisi

    except:

        try:

            varsayilan_fiyat = df['Close'].squeeze().iloc[-1]

            return varsayilan_fiyat, np.full(5, varsayilan_fiyat)

        except:

            return 0.0, np.zeros(5)

# =========================================================
# STREAMLIT PANEL
# =========================================================
if IS_STREAMLIT:

    st.set_page_config(
        page_title="Mobil Borsa",
        layout="centered"
    )

    st.markdown("""
    <style>

    .stApp {
        background-color: #121212;
        color: white;
    }

    div[data-testid="stMetricWidget"] {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 10px;
    }

    </style>
    """, unsafe_allow_html=True)

    st.title("📱 Mobil Borsa")

    db = Veritabani()

    sekme1, sekme2, sekme3 = st.tabs([
        "PORTFÖY",
        "ANALİZ",
        "RADAR"
    ])

    # =====================================================
    # 1. SEKME
    # =====================================================
    with sekme1:

        st.subheader("💼 Portföy")

        with st.expander("➕ Hisse Ekle"):

            yeni_hisse = st.text_input(
                "Hisse Kodu",
                key="mob_ekle_kod"
            ).upper().strip()

            maliyet = st.number_input(
                "Maliyet",
                value=0.0,
                key="mob_ekle_mal"
            )

            adet = st.number_input(
                "Adet",
                value=0,
                key="mob_ekle_adet"
            )

            if st.button("Kaydet / Güncelle"):

                if yeni_hisse:

                    db.hisse_ekle(
                        yeni_hisse,
                        maliyet,
                        adet
                    )

                    st.success(
                        f"{yeni_hisse} kaydedildi"
                    )

                    st.session_state["mob_ekle_kod"] = ""
                    st.session_state["mob_ekle_mal"] = 0.0
                    st.session_state["mob_ekle_adet"] = 0

        hisseler = db.listeyi_getir()

        if not hisseler:

            st.warning("Portföy boş")

        else:

            for h, maliyet, adet in hisseler:

                sorgu_kodu = h + ".IS"

                try:

                    df = yf.download(
                        sorgu_kodu,
                        period="5d",
                        interval="1d",
                        progress=False
                    )

                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)

                    if df.empty:
                        continue

                    fiyat = df['Close'].iloc[-1]

                    if maliyet > 0:

                        degisim = (
                            (fiyat - maliyet)
                            / maliyet
                        ) * 100

                    else:

                        degisim = 0

                    c1, c2 = st.columns([3, 1])

                    c1.metric(
                        h,
                        f"{fiyat:.2f} TL",
                        f"{degisim:+.2f}%"
                    )

                    c1.write(f"Maliyet: {maliyet}")
                    c1.write(f"Adet: {adet}")

                    if c2.button("Sil", key=h):

                        db.hisse_sil(h)

                        st.rerun()

                except Exception as e:

                    st.error(f"{h} hata: {e}")

    # =====================================================
    # 2. SEKME
    # =====================================================
    with sekme2:

        st.subheader("📈 Hisse Analizi")

        hisse_kodu = st.text_input(
            "Hisse Kodu",
            key="analiz_hisse"
        ).upper().strip()

        if hisse_kodu:

            try:

                sorgu_kodu = hisse_kodu + ".IS"

                df = yf.download(
                    sorgu_kodu,
                    period="60d",
                    interval="1d",
                    progress=False
                )

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)

                if df.empty:

                    st.error("Veri bulunamadı")

                else:

                    kapanis = df['Close']

                    son_fiyat = kapanis.iloc[-1]

                    hedef_fiyat, tahmin_serisi = mobil_tahmin_motoru(df)

                    son_rsi = ta.momentum.rsi(
                        kapanis,
                        window=14
                    ).iloc[-1]

                    macd_obj = ta.trend.MACD(kapanis)

                    macd = macd_obj.macd().iloc[-1]

                    macd_sinyal = macd_obj.macd_signal().iloc[-1]

                    if (
                        son_rsi < 42
                        and macd > macd_sinyal
                    ):

                        sinyal = "AL"

                    elif son_rsi > 70:

                        sinyal = "SAT"

                    else:

                        sinyal = "TUT"

                    st.metric(
                        hisse_kodu,
                        f"{son_fiyat:.2f} TL",
                        sinyal
                    )

                    st.write(f"RSI: {son_rsi:.2f}")

                    st.write(
                        f"YZ Tahmin: {hedef_fiyat:.2f} TL"
                    )

                    # GRAFİK
                    fig, ax = plt.subplots(
                        figsize=(6, 3)
                    )

                    ax.plot(
                        kapanis.tail(30).values
                    )

                    ax.grid(True)

                    st.pyplot(fig)

            except Exception as e:

                st.error(f"Hata: {e}")

    # =====================================================
    # 3. SEKME
    # =====================================================
    with sekme3:

        st.subheader("🚀 Mega Radar")

        st.write(
            "AL sinyali veren hisseler taranır."
        )

        if st.button("Taramayı Başlat"):

            bulunanlar = []

            bar = st.progress(0)

            durum = st.empty()

            toplam = len(TUM_BIST)

            for idx, hisse in enumerate(TUM_BIST):

                durum.text(
                    f"Taranıyor: {hisse}"
                )

                oran = (idx + 1) / toplam

                bar.progress(oran)

                try:

                    df = yf.download(
                        hisse + ".IS",
                        period="40d",
                        interval="1d",
                        progress=False
                    )

                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)

                    if df.empty:
                        continue

                    kapanis = df['Close']

                    son_rsi = ta.momentum.rsi(
                        kapanis,
                        window=14
                    ).iloc[-1]

                    macd_obj = ta.trend.MACD(kapanis)

                    macd = macd_obj.macd().iloc[-1]

                    macd_sinyal = macd_obj.macd_signal().iloc[-1]

                    if (
                        son_rsi < 42
                        and macd > macd_sinyal
                    ):

                        bulunanlar.append(hisse)

                except:
                    pass

            bar.empty()

            durum.text("Tarama tamamlandı")

            if bulunanlar:

                st.success(
                    f"{len(bulunanlar)} hisse bulundu"
                )

                st.write(", ".join(bulunanlar))

            else:

                st.warning(
                    "Sinyal veren hisse bulunamadı"
                )


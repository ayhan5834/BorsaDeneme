# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:42:45 2026

@author: EmirAysu
"""



import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import ta

from sklearn.linear_model import HuberRegressor

# ======================================================
# SAYFA AYARI
# ======================================================
st.set_page_config(
    page_title="Mobil Borsa",
    layout="centered"
)

# ======================================================
# CSS
# ======================================================
st.markdown("""
<style>

.stApp{
    background-color:#121212;
    color:white;
}

div[data-testid="stMetricWidget"]{
    background-color:#1E1E1E;
    border:1px solid #2D2D2D;
    padding:10px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# VERİTABANI
# ======================================================
class Veritabani:

    def __init__(self):

        self.conn = sqlite3.connect(
            "takip_listesi.db",
            check_same_thread=False
        )

        self.cur = self.conn.cursor()

        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS watchlist(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hisse TEXT UNIQUE,
            maliyet REAL,
            adet INTEGER
        )
        """)

        self.conn.commit()

    def ekle(self, hisse, maliyet, adet):

        try:

            self.cur.execute("""
            INSERT INTO watchlist
            (hisse,maliyet,adet)
            VALUES(?,?,?)
            """,(hisse,maliyet,adet))

        except:

            self.cur.execute("""
            UPDATE watchlist
            SET maliyet=?,
                adet=?
            WHERE hisse=?
            """,(maliyet,adet,hisse))

        self.conn.commit()

    def sil(self,hisse):

        self.cur.execute("""
        DELETE FROM watchlist
        WHERE hisse=?
        """,(hisse,))

        self.conn.commit()

    def liste(self):

        self.cur.execute("""
        SELECT hisse,maliyet,adet
        FROM watchlist
        """)

        return self.cur.fetchall()

db = Veritabani()

# ======================================================
# TAHMİN MOTORU
# ======================================================
def tahmin_motoru(df):

    try:

        data = df.tail(60).copy()

        data["gun"] = range(len(data))

        X = data[["gun"]]

        y = data["Close"]

        model = HuberRegressor()

        model.fit(X,y)

        son = data["gun"].iloc[-1]

        gelecek = pd.DataFrame({
            "gun":range(son+1,son+6)
        })

        tahmin = model.predict(gelecek)

        return tahmin[-1]

    except:

        return df["Close"].iloc[-1]

# ======================================================
# BAŞLIK
# ======================================================
st.title("📱 Mobil Borsa")

# ======================================================
# TABLAR
# ======================================================
tab1,tab2,tab3 = st.tabs([
    "💼 Portföy",
    "📈 Analiz",
    "🚀 Radar"
])

# ======================================================
# PORTFÖY
# ======================================================
with tab1:

    st.subheader("Portföy Yönetimi")

    with st.expander("➕ Yeni Hisse Ekle"):

        with st.form(
            "ekle_formu",
            clear_on_submit=True
        ):

            hisse = st.text_input(
                "Hisse Kodu"
            )

            hisse = hisse.upper().strip()

            maliyet = st.number_input(
                "Maliyet",
                value=0.0,
                step=0.1
            )

            adet = st.number_input(
                "Adet",
                value=0,
                step=1
            )

            kaydet = st.form_submit_button(
                "Kaydet"
            )

            if kaydet:

                if hisse != "":

                    db.ekle(
                        hisse,
                        maliyet,
                        adet
                    )

                    st.success(
                        f"{hisse} kaydedildi"
                    )

    hisseler = db.liste()

    if len(hisseler) == 0:

        st.warning("Portföy boş")

    else:

        for hisse,maliyet,adet in hisseler:

            try:

                kod = hisse + ".IS"

                df = yf.download(
                    kod,
                    period="5d",
                    interval="1d",
                    progress=False
                )

                if df.empty:
                    continue

                fiyat = float(df["Close"].iloc[-1])

                degisim = 0

                if maliyet > 0:

                    degisim = (
                        (fiyat-maliyet)
                        / maliyet
                    ) * 100

                c1,c2 = st.columns([4,1])

                c1.metric(
                    hisse,
                    f"{fiyat:.2f} TL",
                    f"{degisim:+.2f}%"
                )

                c1.write(f"Maliyet: {maliyet}")
                c1.write(f"Adet: {adet}")

                if c2.button(
                    "Sil",
                    key=hisse
                ):

                    db.sil(hisse)

                    st.success(
                        f"{hisse} silindi"
                    )

                    st.rerun()

            except Exception as e:

                st.error(e)

# ======================================================
# ANALİZ
# ======================================================
with tab2:

    st.subheader("Hisse Analizi")

    analiz_hisse = st.text_input(
        "Hisse Giriniz"
    )

    analiz_hisse = analiz_hisse.upper().strip()

    if analiz_hisse != "":

        try:

            kod = analiz_hisse + ".IS"

            df = yf.download(
                kod,
                period="60d",
                interval="1d",
                progress=False
            )

            if not df.empty:

                kapanis = df["Close"]

                son_fiyat = float(
                    kapanis.iloc[-1]
                )

                rsi = ta.momentum.rsi(
                    kapanis,
                    window=14
                ).iloc[-1]

                macd_obj = ta.trend.MACD(
                    kapanis
                )

                macd = macd_obj.macd().iloc[-1]

                signal = macd_obj.macd_signal().iloc[-1]

                tahmin = tahmin_motoru(df)

                if rsi < 42 and macd > signal:

                    durum = "AL"

                elif rsi > 70:

                    durum = "SAT"

                else:

                    durum = "TUT"

                st.metric(
                    analiz_hisse,
                    f"{son_fiyat:.2f} TL",
                    durum
                )

                st.write(f"RSI: {rsi:.2f}")

                st.write(
                    f"YZ Tahmin: {tahmin:.2f} TL"
                )

                fig,ax = plt.subplots(
                    figsize=(6,3)
                )

                ax.plot(
                    kapanis.tail(30).values
                )

                ax.grid(True)

                st.pyplot(fig)

        except Exception as e:

            st.error(e)

# ======================================================
# RADAR
# ======================================================
with tab3:

    st.subheader("Mega Radar")

    bist = [
        "THYAO",
        "ASELS",
        "KRDMD",
        "SASA",
        "HEKTS",
        "SISE",
        "AKBNK",
        "TUPRS",
        "EREGL"
    ]

    if st.button("Taramayı Başlat"):

        bulunan = []

        progress = st.progress(0)

        durum = st.empty()

        toplam = len(bist)

        for i,hisse in enumerate(bist):

            durum.text(
                f"Taranıyor: {hisse}"
            )

            progress.progress(
                (i+1)/toplam
            )

            try:

                df = yf.download(
                    hisse + ".IS",
                    period="40d",
                    interval="1d",
                    progress=False
                )

                if df.empty:
                    continue

                kapanis = df["Close"]

                rsi = ta.momentum.rsi(
                    kapanis,
                    window=14
                ).iloc[-1]

                macd_obj = ta.trend.MACD(
                    kapanis
                )

                macd = macd_obj.macd().iloc[-1]

                signal = macd_obj.macd_signal().iloc[-1]

                if rsi < 42 and macd > signal:

                    bulunan.append(hisse)

            except:
                pass

        progress.empty()

        if len(bulunan) > 0:

            st.success(
                f"{len(bulunan)} hisse bulundu"
            )

            st.write(
                ", ".join(bulunan)
            )

        else:

            st.warning(
                "Sinyal bulunamadı"
            )



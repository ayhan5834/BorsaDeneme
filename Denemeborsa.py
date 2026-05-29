# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:42:45 2026

@author: EmirAysu
"""
import os
import sys
import logging

# Matplotlib arkada harici pencere açmasını engeller ve logları kapatır
import matplotlib
matplotlib.use('Agg') 
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# STANDART KÜTÜPHANELER
import sqlite3
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import yfinance as yf
import ta
from sklearn.linear_model import HuberRegressor

# ==============================================================================
# 1. VERİTABANI SINIFI (PORTFÖYÜ TELEFONDAN YÖNETMEK İÇİN)
# ==============================================================================
class Veritabani:
    def __init__(self):
        self.baglanti = sqlite3.connect("takip_listesi.db", check_same_thread=False)
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
# 2. DİNAMİK BIST LİSTESİ MOTORU (YENİ HALKA ARZLAR DAHİL)
# ==============================================================================
def dinamik_bist_listesi_yukle():
    csv_yolu = "bist_hisseler.csv"
    url = "https://raw.githubusercontent.com/atas/borsa-istanbul-hisse-listesi/main/bist_hisseler.csv"
    
    try:
        df_canli = pd.read_csv(url, timeout=5)
        if "kod" in df_canli.columns and not df_canli.empty:
            df_canli.to_csv(csv_yolu, index=False)
            return df_canli["kod"].tolist()
    except:
        pass

    try:
        if os.path.exists(csv_yolu):
            df_yerel = pd.read_csv(csv_yolu)
            return df_yerel["kod"].tolist()
    except:
        pass

    # İnternet veya dosya erişiminde sorun olursa acil durum yedek listesi
    return ["AKBNK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "EREGL", "FROTO", "GARAN", "ISCTR", "KCHOL", "THYAO", "TUPRS", "YKBNK"]

TUM_BIST = dinamik_bist_listesi_yukle()

# ==============================================================================
# 3. YAPAY ZEKA TAHMİN MOTORU (HUBER REGRESSOR)
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
        varsayilan_fiyat = df['Close'].squeeze().iloc[-1]
        return varsayilan_fiyat, np.full(5, varsayilan_fiyat)

# ==============================================================================
# 4. STREAMLIT MOBİL ARAYÜZ TASARIMI
# ==============================================================================
import streamlit as st
st.set_page_config(page_title="Mobil Borsa", layout="centered")

# Karanlık ve şık mobil tema ayarı
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #FFFFFF; }
    div[data-testid="stExpander"] { background-color: #1E1E1E; border: 1px solid #2D2D2D; border-radius: 10px; }
    div[data-testid="stMetricWidget"] { background-color: #1E1E1E; border: 1px solid #2D2D2D; padding: 10px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("📱 Mobil Borsa Paneli")
db = Veritabani()

sekme1, sekme2, sekme3 = st.tabs(["PORTFÖY & STOP", "HİSSE ANALİZ", "MEGA RADAR"])

# --- 1. SEKME: PORTFÖY VE KASA DURUMU ---
with sekme1:
    st.subheader("💼 Portföy & Durum")
    hisseler = db.listeyi_getir()
    
    with st.expander("➕ Yeni Hisse Ekle / Maliyet Düzenle"):
        yeni_hisse = st.text_input("Hisse Kodu (örn: ASELS)", key="mob_ekle_kod").upper().strip()
        maliyet = st.number_input("Maliyet", value=0.0, step=0.1, key="mob_ekle_mal")
        adet = st.number_input("Adet", value=0, step=1, key="mob_ekle_adet")
        if st.button("Kaydet / Güncelle", key="mob_kaydet_btn"):
            if yeni_hisse:
                db.hisse_ekle(yeni_hisse, maliyet, adet)
                st.success(f"{yeni_hisse} portföye kaydedildi!")
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
                df = yf.download(sorgu_kodu, period="2d", interval="1d", progress=False)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                
                if df is None or df.empty:
                    kartlar_verisi.append((h, 0.0, "Veri Bulunamadı", adet, 0.0, "KOD HATALI", "#FF9800"))
                    continue
                    
                bugun_fiyat = df['Close'].squeeze().iloc[-1]
                if maliyet > 0:
                    degisim = ((bugun_fiyat - maliyet) / maliyet) * 100
                    toplam_maliyet_hacmi += (maliyet * adet)
                    toplam_guncel_hacim += (bugun_fiyat * adet)
                    maliyet_metni = f"Maliyet: {maliyet:.2f} TL"
                else:
                    dun_fiyat = df['Close'].squeeze().iloc[-2] if len(df) >= 2 else bugun_fiyat
                    degisim = ((bugun_fiyat - dun_fiyat) / dun_fiyat) * 100
                    maliyet_metni = "Takip"
                
                # Akıllı stop koruma sinyalleri
                if maliyet > 0 and degisim <= -5.0: status, renk = "🚨 STOP!!", "#E74C3C"
                elif maliyet > 0 and degisim <= -3.0: status, renk = "⚠️ STP.UYARI", "#E67E22"
                elif maliyet > 0 and degisim >= 10.0: status, renk = "🟢 KÂR AL", "#2ECC71"
                elif degisim > 0: status, renk = "📈 YÜKSELİŞ", "#27AE60"
                else: status, renk = "📉 DÜŞÜŞ", "#C0392B"
                
                kartlar_verisi.append((h, bugun_fiyat, maliyet_metni, adet, degisim, status, renk))
            except:
                kartlar_verisi.append((h, 0.0, "Bağlantı Yok", adet, 0.0, "HATA", "#FF9800"))
        
        if toplam_maliyet_hacmi > 0:
            toplam_kar_zarar_yuzde = ((toplam_guncel_hacim - toplam_maliyet_hacmi) / toplam_maliyet_hacmi) * 100
            st.markdown(f"""
            <div style='background-color: #1E1E1E; padding: 12px; border-radius: 10px; border: 1px solid #2D2D2D; text-align: center;'>
                <span style='color: #00F0FF; font-weight: bold; font-size: 16px;'>
                    Kasa: {toplam_maliyet_hacmi:,.2f} TL → Net Durum: %{toplam_kar_zarar_yuzde:+,.2f}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
        for h, fiyat, m_metni, adet, degisim, status, renk in kartlar_verisi:
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 2, 1])
                c1.metric(label=f"{h} ({status})", value=f"{fiyat:.2f} TL" if fiyat > 0 else "N/A", delta=f"{degisim:+.2f}%" if fiyat > 0 else None)
                c2.write(f"**{m_metni}**")
                c2.write(f"Adet: {adet}")
                if c3.button("🗑️ Sil", key=f"del_{h}"):
                    db.hisse_sil(h)
                    st.rerun()

# --- 2. SEKME: DETAYLI HİSSE ANALİZİ VE GRAFİK ---
with sekme2:
    st.subheader("🔍 Detaylı Hisse Analizi")
    hisse_kodu = st.text_input("Hisse Kodu Giriniz (Örn: THYAO)", key="mob_analiz_input").upper().strip()
    
    if hisse_kodu:
        sorgu_kodu = hisse_kodu if hisse_kodu.endswith(".IS") else hisse_kodu + ".IS"
        try:
            df = yf.download(sorgu_kodu, period="60d", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            
            if df is None or df.empty or len(df) < 5:
                st.error("Hisse verisi bulunamadı.")
            else:
                kapanis = df['Close'].squeeze()
                son_fiyat = kapanis.iloc[-1]
                
                hedef_fiyat, tahmin_serisi = mobil_tahmin_motoru(df)
                potansiyel_getiri = ((hedef_fiyat - son_fiyat) / son_fiyat) * 100 if son_fiyat > 0 else 0
                
                son_rsi = ta.momentum.rsi(kapanis, window=14).iloc[-1]
                macd_obj = ta.trend.MACD(kapanis)
                macd_cizgisi = macd_obj.macd().iloc[-1]
                macd_sinyal = macd_obj.macd_signal().iloc[-1]
                
                if (son_rsi < 42 and macd_cizgisi > macd_sinyal) or (son_rsi < 30):
                    genel_durum, s_renk = "AL", "#2ECC71"
                elif (son_rsi > 70) or (macd_cizgisi < macd_sinyal):
                    genel_durum, s_renk = "SAT", "#E74C3C"
                else:
                    genel_durum, s_renk = "TUT", "#8A8A8A"
                    
                st.markdown(f"""
                <div style='background-color: #1E1E1E; padding: 20px; border-radius: 15px; border: 1px solid #2D2D2D; text-align: center; margin-bottom: 15px;'>
                    <h2 style='margin: 0; color: white;'>{hisse_kodu}</h2>
                    <h1 style='margin: 10px 0; color: #00F0FF;'>{son_fiyat:,.2f} TL</h1>
                    <div style='background-color: {s_renk}; color: #121212; padding: 6px; border-radius: 8px; font-weight: bold; display: inline-block; width: 100%;'>
                        {genel_durum}
                    </div>
                    <p style='margin-top: 10px; font-size: 14px; color: white;'>RSI (14): {son_rsi:.2f}</p>
                    <p style='color: #8A8A8A; font-size: 13px;'>🚀 YZ 5 Günlük Tahmin: <b>{hedef_fiyat:.2f} TL</b> (Potansiyel: %{potansiyel_getiri:+.2f})</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Grafik Alanı
                fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#121212')
                ax.set_facecolor('#1E1E1E')
                
                kapanislar_son30 = kapanis.tail(30)
                gunler = np.arange(len(kapanislar_son30))
                
                ax.plot(gunler, kapanislar_son30.values, color='#00F0FF', linewidth=2, label="Gerçek")
                ax.fill_between(gunler, kapanislar_son30.values, min(kapanislar_son30.values)*0.99, color='#00F0FF', alpha=0.08)
                
                tahmin_gunler = np.arange(len(kapanislar_son30)-1, len(kapanislar_son30) + 4)
                tahmin_degerleri = np.concatenate(([kapanislar_son30.iloc[-1]], tahmin_serisi))
                ax.plot(tahmin_gunler, tahmin_degerleri, color='#FF00FF', linestyle='--', linewidth=2, label="YZ Tahmin")
                
                ax.tick_params(colors='white', labelsize=8)
                ax.grid(True, color='#2D2D2D', linestyle='--')
                ax.legend(loc='upper left', fontsize=8, facecolor='#1E1E1E', labelcolor='white')
                for spine in ax.spines.values(): spine.set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Analiz hatası: {e}")

# --- 3. SEKME: MEGA RADAR TARAMASI (CANLI LİSTEDEN) ---
with sekme3:
    st.subheader("🔍 Mega Radar Taraması")
    st.write("Canlı indirilen tüm BIST listesi taranarak AL sinyali verenler filtrelenir.")
    
    if st.button("🚀 TÜM BORSAYI TARAMAYA BAŞLAT", key="mob_radar_start"):
        bulunanlar = []
        ilerleme_bari = st.progress(0)
        durum_alani = st.empty()
        
        toplam = len(TUM_BIST)
        for idx, h in enumerate(TUM_BIST):
            durum_alani.text(f"Taranıyor: {h} ({idx+1}/{toplam})")
            ilerleme_bari.progress((idx + 1) / toplam)
            try:
                df = yf.download(h + ".IS", period="40d", interval="1d", progress=False)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                
                if df is None or df.empty or len(df) < 15: continue
                
                kapanis = df['Close'].squeeze()
                son_rsi = ta.momentum.rsi(kapanis, window=14).iloc[-1]
                macd_obj = ta.trend.MACD(kapanis)
                macd_cizgisi = macd_obj.macd().iloc[-1]
                macd_sinyal = macd_obj.macd_signal().iloc[-1]
                
                if (son_rsi < 40 and macd_cizgisi > macd_sinyal) or (son_rsi < 30):
                    fiyat = kapanis.iloc[-1]
                    bulunanlar.append(f"🟢 **{h}** → Fiyat: {fiyat:.2f} TL (RSI: {son_rsi:.1f}) -> **AL SİNYALİ**")
            except:
                continue
        
        durum_alani.success(f"Tarama Tamamlandı! Toplam {len(bulunanlar)} adet fırsat yakalandı.")
        for b in bulunanlar: st.markdown(b)

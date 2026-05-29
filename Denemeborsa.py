# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:42:45 2026

@author: EmirAysu
"""

import os
import sys
import logging

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
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import yfinance as yf
import ta
from sklearn.linear_model import HuberRegressor
import streamlit as st

IS_STREAMLIT = "streamlit" in sys.modules

# ==============================================================================
# 1. VERİTABANI SINIFI
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

    def hisse_detay_getir(self, kod):
        self.cursor.execute("SELECT maliyet, adet FROM watchlist WHERE hisse_kodu = ?", (kod,))
        return self.cursor.fetchone()

# ==============================================================================
# 2. DİNAMİK BIST LİSTESİ MOTORU (HALKA ARZLAR DAHİL)
# ==============================================================================
@st.cache_data(ttl=3600)
def dinamik_bist_listesi_yukle():
    csv_yolu = "bist_hisseler.csv"
    # Eğer GitHub'a yüklediyseniz, sadece dosya ismini kontrol etmek yeterlidir
    if os.path.exists(csv_yolu):
        df = pd.read_csv(csv_yolu)
        return df["kod"].tolist()
    
    # Dosya yoksa yedek listeye dön
    return ["A1CAP", "ADEL", "AGROT", "AKBNK", "ALARK", ...]

# Canlı listeyi değişkene aktar
TUM_BIST = dinamik_bist_listesi_yukle()

# ==============================================================================
# 3. YAPAY ZEKA TAHMİN MOTORU (BOŞ VERİ KORUMALI)
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
# 4. STREAMLIT MOBİL UYGULAMA PANELİ
# ==============================================================================
if IS_STREAMLIT:    
    import streamlit as st
    st.set_page_config(page_title="Mobil Borsa", layout="centered")
    
    st.markdown("""
        <style>
        .stApp { background-color: #121212; color: #FFFFFF; }
        div[data-testid="stExpander"] { background-color: #1E1E1E; border: 1px solid #2D2D2D; border-radius: 10px; }
        div[data-testid="stMetricWidget"] { background-color: #1E1E1E; border: 1px solid #2D2D2D; padding: 10px; border-radius: 10px; }
        /* Sadece Buton Renkleri Koyu Mavi */
        div.stButton > button { background-color: #00008B !important; color: white !important; }
        </style>
    """, unsafe_allow_html=True)

    st.title("📱 Mobil Borsa")
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
                    
                    if df is None or df.empty or len(df) == 0:
                        kartlar_verisi.append((h, 0.0, "Veri Bulunamadı", adet, 0.0, "KOD HATALI / DELISTED", "#FF9800"))
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
            else:
                st.info("Maliyet girilmemiş takip hisseleri.")
                
            st.write("")
            
            for h, fiyat, m_metni, adet, degisim, status, renk in kartlar_verisi:
                with st.container(border=True):
                    # Kolon oranlarını biraz değiştirdik: c1 (Fiyat/Durum), c2 (Detaylar), c3 (Sil Butonu)
                    c1, c2, c3 = st.columns([2.5, 2.5, 1])
                    
                    # Fiyat ve Delta bilgisi
                    c1.metric(label=f"{h} ({status})", value=f"{fiyat:.2f} TL" if fiyat > 0 else "N/A", 
                              delta=f"{degisim:+.2f}%" if fiyat > 0 else None)
                    
                    # Maliyet ve Adet bilgilerini yan yana koymak için yeni bir column yapısı
                    sub_c1, sub_c2 = c2.columns(2)
                    sub_c1.write(f"**{m_metni}**")
                    sub_c2.write(f"**Adet:** {adet}")
                    
                    # Silme butonu
                    if c3.button("🗑️ Sil", key=f"del_{h}"):
                        db.hisse_sil(h)
                        st.rerun()
                        
        if st.button("🔄 Verileri Yenile", key="mob_global_yenile"):
            st.rerun()

    # --- 2. SEKME: PANEL KART ANALİZİ (YZ + GRAFİK + HACİM) ---
    with sekme2:
        st.subheader("🔍 Detaylı Hisse Analizi")
        hisse_kodu = st.text_input("Hisse Kodu (Örn: THYAO)", key="mob_analiz_input").upper().strip()
        
        if hisse_kodu:
            sorgu_kodu = hisse_kodu if hisse_kodu.endswith(".IS") else hisse_kodu + ".IS"
            try:
                df = yf.download(sorgu_kodu, period="60d", interval="1d", progress=False)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                
                if df is not None and not df.empty:
                    kapanis = df['Close'].squeeze()
                    hacim = df['Volume'].squeeze()
                    son_fiyat = kapanis.iloc[-1]
                    
                    # YZ TAHMİNİ
                    hedef_fiyat, tahmin_serisi = mobil_tahmin_motoru(df)
                    potansiyel = ((hedef_fiyat - son_fiyat) / son_fiyat) * 100
                    
                    # HACİM ONAYI
                    hacim_ort = hacim.rolling(10).mean().iloc[-1]
                    hacim_onay = hacim.iloc[-1] > (hacim_ort * 0.8)
                    
                    st.markdown(f"""
                    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 1px solid #2D2D2D;'>
                        <h3 style='color: white;'>{hisse_kodu} Analizi</h3>
                        <p>Fiyat: <b>{son_fiyat:,.2f} TL</b> | Hacim Onay: {'✅' if hacim_onay else '❌'}</p>
                        <p style='color: #00F0FF;'>🚀 YZ 5 Günlük Tahmin: <b>{hedef_fiyat:.2f} TL</b> (%{potansiyel:+.2f})</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # --- GÜVENLİ GRAFİK ÇİZİMİ ---
                    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#121212')
                    ax.set_facecolor('#1E1E1E')
                    
                    kapanislar_son30 = kapanis.tail(30)
                    gunler = np.arange(len(kapanislar_son30))
                    
                    # Gerçek veri
                    ax.plot(gunler, kapanislar_son30.values, color='#00F0FF', linewidth=2, label="Gerçek")
                    ax.fill_between(gunler, kapanislar_son30.values, min(kapanislar_son30.values)*0.99, color='#00F0FF', alpha=0.08)
                    
                    # Tahmin verisi (Boyutları eşitleyen dinamik yapı)
                    son_gercek_gun = gunler[-1]
                    tahmin_gunler = np.arange(son_gercek_gun, son_gercek_gun + len(tahmin_serisi) + 1)
                    tahmin_degerleri = np.concatenate(([kapanislar_son30.iloc[-1]], tahmin_serisi))
                    
                    # Dizilerin boyutu 1 tane bile farklı olsa hata almamak için eşitleme (Safety Check)
                    min_len = min(len(tahmin_gunler), len(tahmin_degerleri))
                    ax.plot(tahmin_gunler[:min_len], tahmin_degerleri[:min_len], color='#FF00FF', linestyle='--', linewidth=2, label="YZ Tahmin")
                    
                    ax.tick_params(colors='white', labelsize=8)
                    ax.grid(True, color='#2D2D2D', linestyle='--')
                    ax.legend(loc='upper left', fontsize=8, facecolor='#1E1E1E', labelcolor='white')
                    for spine in ax.spines.values(): spine.set_visible(False)
                    fig.tight_layout()
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Analiz hatası: {e}")

    # --- 3. SEKME: MEGA RADAR TARAMASI (GÜNCELLENMİŞ VE İLERLEME ÇUBUKLU) ---
    with sekme3:
        st.subheader("🔍 Radar Taraması")
        
        col1, col2 = st.columns(2)
        hacim_filtresi = col1.checkbox("Hacim Onayı İstiyorum", value=True)
        sadece_guclu = col2.checkbox("Sadece GÜÇLÜ AL Sinyalleri", value=False)
        
        if st.button("🚀 TARAMAYI BAŞLAT", key="mob_radar_start"):
            guncel_hisse_listesi = dinamik_bist_listesi_yukle()
            bulunanlar = []
            toplam = len(guncel_hisse_listesi)
            
            # İlerleme elemanlarını hazırla
            ilerleme_bari = st.progress(0)
            durum_alani = st.empty()
            
            for idx, h in enumerate(guncel_hisse_listesi):
                # Anlık sayaç ve bilgi güncelleme
                durum_alani.text(f"Taranıyor: {h} ({idx+1}/{toplam})")
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
                except: 
                    continue
            
            # İşlem bittiğinde temizle
            durum_alani.text("Tarama tamamlandı!")
            ilerleme_bari.empty()
            
            if bulunanlar:
                st.success(f"✅ {len(bulunanlar)} adet hisse kriterlerine uygun:")
                for hisse in bulunanlar: st.markdown(f"🔹 **{hisse}**")
            else:
                st.warning("Seçili kriterlerde hisse bulunamadı.")

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
def dinamik_bist_listesi_yukle():
    """
    Yeni halka arzları ve güncel hisseleri GitHub üzerinden canlı çeker.
    İnternet yoksa yerel CSV'ye bakar, o da yoksa BIST30 çekirdek listesine döner.
    """
    csv_yolu = "bist_hisseler.csv"
    url = "https://raw.githubusercontent.com/atas/borsa-istanbul-hisse-listesi/main/bist_hisseler.csv"
    
    # 1. Adım: Canlı güncel listeyi internetten çekmeyi dene
    try:
        print("Güncel BIST Listesi internetten indiriliyor (Halka arzlar dahil)...")
        df_canli = pd.read_csv(url, timeout=5)
        if "kod" in df_canli.columns and not df_canli.empty:
            # İleride internet olmadığında kullanmak üzere yerel diskte güncelle/yedekle
            df_canli.to_csv(csv_yolu, index=False)
            print(f"Başarılı! {len(df_canli)} hisse güncellendi.")
            return df_canli["kod"].tolist()
    except Exception as e:
        print(f"Canlı liste alınamadı (İnternet hatası/Zaman aşımı): {e}. Yerel kaynak deneniyor...")

    # 2. Adım: İnternet başarısızsa önceden kaydedilmiş yerel CSV dosyasını oku
    try:
        if os.path.exists(csv_yolu):
            df_yerel = pd.read_csv(csv_yolu)
            if "kod" in df_yerel.columns and not df_yerel.empty:
                print(f"Yerel CSV başarıyla okundu: {len(df_yerel)} hisse yüklendi.")
                return df_yerel["kod"].tolist()
    except Exception as e:
        print(f"Yerel CSV okuma hatası: {e}")

    # 3. Adım: Tamamen internetsiz ve ilk açılışsa uygulamanın çökmemesi için acil durum kemik listesi
    print("Yedek sabit hisse listesi devreye alınıyor...")
    return [
        "A1CAP", "ADEL", "AGROT", "AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", 
        "BRSAN", "CCOLA", "CIMSA", "DOAS", "DOHOL", "EKGYO", "ENJSA", "ENKAI", "EREGL", 
        "FROTO", "GARAN", "GUBRF", "HALKB", "HEKTS", "ISCTR", "KCAER", "KCHOL", "KONTR", 
        "KOZAL", "MGROS", "MIATK", "ODAS", "OYAKC", "PGSUS", "REEDR", "SAHOL", "SASA", 
        "SISE", "SOKM", "TCELL", "THYAO", "TKFEN", "TOASO", "TUPRS", "VAKBN", "VESTL", "YKBNK"
    ]

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
                    c1, c2, c3 = st.columns([2, 2, 1])
                    c1.metric(label=f"{h} ({status})", value=f"{fiyat:.2f} TL" if fiyat > 0 else "N/A", delta=f"{degisim:+.2f}%" if fiyat > 0 else None)
                    c2.write(f"**{m_metni}**")
                    c2.write(f"Adet: {adet}")
                    if c3.button("🗑️ Sil", key=f"del_{h}"):
                        db.hisse_sil(h)
                        st.rerun()
                        
        if st.button("🔄 Verileri Yenile", key="mob_global_yenile"):
            st.rerun()

    # --- 2. SEKME: PANEL KART ANALİZİ ---
    with sekme2:
        st.subheader("🔍 Detaylı Hisse Analizi")
        hisse_kodu = st.text_input("Hisse Kodu Giriniz (Örn: THYAO)", key="mob_analiz_input").upper().strip()
        
        if hisse_kodu:
            sorgu_kodu = hisse_kodu if hisse_kodu.endswith(".IS") else hisse_kodu + ".IS"
            try:
                df = yf.download(sorgu_kodu, period="60d", interval="1d", progress=False)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                
                if df is None or df.empty or len(df) < 5:
                    st.error("Hisse verisi bulunamadı veya hisse işleme kapalı (Delisted).")
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
                    
                    if st.button("➕ PORTFÖYÜME / LİSTEME EKLE", key="mob_analizden_ekle"):
                        db.hisse_ekle(hisse_kodu, 0.0, 0)
                        st.success(f"{hisse_kodu} listeye eklendi!")
                    
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

    # --- 3. SEKME: MEGA RADAR TARAMASI (YENİ HALKA ARZLAR DAHİL) ---
    with sekme3:
        st.subheader("🔍 Mega Radar Taraması")
        st.write("Tüm BIST hisseleri taranarak AL sinyali üretenler listelenir.")
        
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
                    
                    if df is None or df.empty or len(df) < 15: 
                        continue
                    
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
            for b in bulunanlar:
                st.markdown(b)

# ==============================================================================
# 5. MASAÜSTÜ (PYQT5) UYGULAMA BLOKLARI
# ==============================================================================
if not IS_STREAMLIT:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QPushButton, QLineEdit, QLabel, 
                                 QTabWidget, QListWidget, QListWidgetItem, QStatusBar, 
                                 QMessageBox, QFrame, QGraphicsDropShadowEffect, QInputDialog)
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QFont, QColor

    class BorsaMobilUygulama(QMainWindow):
        def __init__(self):
            super().__init__()
            self.db = Veritabani()
            self.initUI()
            
            self.timer = QTimer()
            self.timer.timeout.connect(self.canli_fiyat_guncelle)
            self.timer.start(10000) 
                
        def yerel_ip_bul(self):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('8.8.8.8', 80))
                ip = s.getsockname()[0]
            except:
                ip = "127.0.0.1"
            finally:
                s.close()
            return ip

        def initUI(self):
            self.setWindowTitle("BIST Kasa Koruyucu")
            self.setGeometry(100, 100, 400, 600)
            
            self.setStyleSheet("""
                QMainWindow { background-color: #121212; }
                QLabel { color: #FFFFFF; }
                QTabWidget::pane { border: none; background-color: #121212; }
                QTabBar::tab {
                    background-color: #1E1E1E;
                    color: #8A8A8A;
                    padding: 12px;
                    font-weight: bold;
                    font-size: 11px;
                    border: none;
                }
                QTabBar::tab:selected {
                    background-color: #121212;
                    color: #00F0FF;
                    border-bottom: 3px solid #00F0FF;
                }
            """)
            
            self.tabs = QTabWidget()
            self.tabs.setTabPosition(QTabWidget.South)
            self.setCentralWidget(self.tabs)
            
            self.tab_watchlist = QWidget()
            self.tab_analiz = QWidget()
            self.tab_tarayici = QWidget()
            
            self.tabs.addTab(self.tab_watchlist, "PORTFÖY & STOP")
            self.tabs.addTab(self.tab_analiz, "HİSSE ANALİZ")
            self.tabs.addTab(self.tab_tarayici, "MEGA RADAR")
            
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            self.grafik_fig = Figure(figsize=(4, 2), dpi=100, facecolor='#121212')
            self.grafik_ekrani = FigureCanvas(self.grafik_fig)
            
            self.kur_watchlist_sekmesi()
            self.kur_analiz_sekmesi()
            self.kur_tarayici_sekmesi()
            
            self.tabs.currentChanged.connect(self.sekme_degisti)

            self.setStatusBar(QStatusBar())
            port = 8501 
            self.lbl_ip_bilgi = QLabel(f"Bağlantı: http://{self.yerel_ip_bul()}:{port}")
            self.lbl_ip_bilgi.setStyleSheet("color: #00F0FF; font-size: 10px; font-weight: bold;")
            self.statusBar().addWidget(self.lbl_ip_bilgi)

        def kur_watchlist_sekmesi(self):
            layout = QVBoxLayout(self.tab_watchlist)
            layout.setContentsMargins(12, 12, 12, 12)
            
            self.lbl_toplam_durum = QLabel("Portföy Durumu Hesaplanıyor...")
            self.lbl_toplam_durum.setFont(QFont("Arial", 11, QFont.Bold))
            self.lbl_toplam_durum.setStyleSheet("color: #00F0FF; background-color: #1E1E1E; padding: 10px; border-radius: 10px; border: 1px solid #2D2D2D;")
            self.lbl_toplam_durum.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.lbl_toplam_durum)
            
            self.liste_widget = QListWidget()
            self.liste_widget.setStyleSheet("""
                QListWidget {
                    background-color: #1E1E1E;
                    border: 1px solid #2D2D2D;
                    border-radius: 15px;
                    padding: 5px;
                    color: white;
                }
                QListWidget::item {
                    padding: 10px;
                    border-bottom: 1px solid #2D2D2D;
                }
                QListWidget::item:selected {
                    background-color: #333333;
                    border-radius: 8px;
                }
            """)
            self.liste_widget.itemDoubleClicked.connect(self.listeden_analize_gonder)
            layout.addWidget(self.liste_widget)
            
            btn_layout = QHBoxLayout()
            self.btn_sil = QPushButton("HİSSE SİL")
            self.btn_sil.setFont(QFont("Arial", 9, QFont.Bold))
            self.btn_sil.setStyleSheet("QPushButton { background-color: #C0392B; color: white; border-radius: 10px; padding: 10px; }")
            self.btn_sil.clicked.connect(self.listeden_hisse_sil)
            
            self.btn_maliyet_duzelt = QPushButton("MALİYET GİR")
            self.btn_maliyet_duzelt.setFont(QFont("Arial", 9, QFont.Bold))
            self.btn_maliyet_duzelt.setStyleSheet("QPushButton { background-color: #00838F; color: white; border-radius: 10px; padding: 10px; }")
            self.btn_maliyet_duzelt.clicked.connect(self.maliyet_guncelle_penceresi)
            
            self.btn_yenile = QPushButton("YENİLE")
            self.btn_yenile.setFont(QFont("Arial", 9, QFont.Bold))
            self.btn_yenile.setStyleSheet("QPushButton { background-color: #2D2D2D; color: white; border-radius: 10px; padding: 10px; border: 1px solid #444; }")
            self.btn_yenile.clicked.connect(self.watchlist_yukle)
            
            btn_layout.addWidget(self.btn_sil)
            btn_layout.addWidget(self.btn_maliyet_duzelt)
            btn_layout.addWidget(self.btn_yenile)
            layout.addLayout(btn_layout)
            
            self.watchlist_yukle()

        def kur_analiz_sekmesi(self):
            layout = QVBoxLayout(self.tab_analiz)
            layout.setContentsMargins(15, 15, 15, 15)
            layout.setSpacing(10)
            
            arama_layout = QHBoxLayout()
            self.hisse_input = QLineEdit()
            self.hisse_input.setPlaceholderText("Hisse Kodu (Örn: THYAO)")
            self.hisse_input.setStyleSheet("QLineEdit { background-color: #1E1E1E; color: white; border: 2px solid #333; border-radius: 10px; padding: 8px; }")
            self.hisse_input.returnPressed.connect(self.hisse_analiz_et)
            
            self.btn_analiz = QPushButton("ANALİZ")
            self.btn_analiz.setStyleSheet("QPushButton { background-color: #00F0FF; color: #121212; font-weight:bold; border-radius: 10px; padding: 9px 12px; }")
            self.btn_analiz.clicked.connect(self.hisse_analiz_et)
            
            arama_layout.addWidget(self.hisse_input, 7)
            arama_layout.addWidget(self.btn_analiz, 3)
            layout.addLayout(arama_layout)
            
            self.btn_takibe_ekle = QPushButton("+ PORTFÖYÜME / LİSTEME EKLE")
            self.btn_takibe_ekle.setFont(QFont("Arial", 9, QFont.Bold))
            self.btn_takibe_ekle.setStyleSheet("QPushButton { background-color: #2E7D32; color: white; border-radius: 10px; padding: 8px; }")
            self.btn_takibe_ekle.clicked.connect(self.listeye_hisse_ekle)
            layout.addWidget(self.btn_takibe_ekle)
            
            self.kart_frame = QFrame()
            self.kart_frame.setStyleSheet("QFrame { background-color: #1E1E1E; border-radius: 15px; border: 1px solid #2D2D2D; }")
            
            golge = QGraphicsDropShadowEffect()
            golge.setBlurRadius(10)
            golge.setColor(QColor(0, 0, 0, 150))
            self.kart_frame.setGraphicsEffect(golge)
            
            kart_layout = QVBoxLayout(self.kart_frame)
            
            self.lbl_hisse_adi = QLabel("Hisse Seçiniz")
            self.lbl_hisse_adi.setFont(QFont("Arial", 16, QFont.Bold))
            self.lbl_hisse_adi.setAlignment(Qt.AlignCenter)
            
            self.lbl_fiyat = QLabel("- TL")
            self.lbl_fiyat.setFont(QFont("Arial", 22, QFont.Bold))
            self.lbl_fiyat.setAlignment(Qt.AlignCenter)
            
            self.lbl_sinyal = QLabel("BEKLENİYOR")
            self.lbl_sinyal.setAlignment(Qt.AlignCenter)
            self.lbl_sinyal.setStyleSheet("background-color: #2D2D2D; color: #AAA; border-radius: 8px; padding: 6px; font-weight: bold;")
            
            self.lbl_rsi = QLabel("RSI (14): -")
            self.lbl_detay = QLabel("Analiz sonuçları burada listelenir.")
            self.lbl_detay.setWordWrap(True)
            self.lbl_detay.setStyleSheet("color: #8A8A8A; font-size: 11px;")
            self.lbl_detay.setAlignment(Qt.AlignCenter)
            
            kart_layout.addWidget(self.lbl_hisse_adi)
            kart_layout.addWidget(self.lbl_fiyat)
            kart_layout.addWidget(self.lbl_sinyal)
            kart_layout.addSpacing(5)
            kart_layout.addWidget(self.lbl_rsi)
            kart_layout.addWidget(self.lbl_detay)
            
            layout.addWidget(self.kart_frame)
            layout.addWidget(self.grafik_ekrani)

        def kur_tarayici_sekmesi(self):
            layout = QVBoxLayout(self.tab_tarayici)
            layout.setContentsMargins(15, 15, 15, 15)
            layout.setSpacing(12)
            
            info_lbl = QLabel("📱 Canlı Yayın & Radar Merkezi")
            info_lbl.setFont(QFont("Arial", 13, QFont.Bold))
            info_lbl.setAlignment(Qt.AlignCenter)
            info_lbl.setStyleSheet("color: #00F0FF;")
            layout.addWidget(info_lbl)
            
            self.btn_taramayi_baslat = QPushButton("🔍 MASAÜSTÜ RADAR TARAMASI BAŞLAT")
            self.btn_taramayi_baslat.setFont(QFont("Arial", 9, QFont.Bold))
            self.btn_taramayi_baslat.setStyleSheet("QPushButton { background-color: #2D2D2D; color: white; border-radius: 10px; padding: 10px; border: 1px solid #444; }")
            self.btn_taramayi_baslat.clicked.connect(self.mega_taramayi_baslat)
            layout.addWidget(self.btn_taramayi_baslat)

            self.lbl_radar_durum = QLabel("Masaüstü radarı hazır (Canlı listeden besleniyor).")
            self.lbl_radar_durum.setStyleSheet("color: #8A8A8A; font-size: 10px;")
            self.lbl_radar_durum.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.lbl_radar_durum)

            self.tarayici_liste = QListWidget()
            self.tarayici_liste.setStyleSheet("QListWidget { background-color: #1E1E1E; border: 1px solid #2D2D2D; border-radius: 10px; color: white; }")
            self.tarayici_liste.itemDoubleClicked.connect(self.tarayicidan_analize_gonder)
            layout.addWidget(self.tarayici_liste)
            
            self.btn_mobil_yayin = QPushButton("🚀 MOBİL YAYINI BAŞLAT")
            self.btn_mobil_yayin.setFont(QFont("Arial", 10, QFont.Bold))
            self.btn_mobil_yayin.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #11998e, stop:1 #38ef7d);
                    color: white;
                    border-radius: 12px;
                    padding: 12px;
                    border: 1px solid #27ae60;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #15ad9f, stop:1 #4efc8f);
                }
            """)
            self.btn_mobil_yayin.clicked.connect(self.mobil_yayini_baslat)
            layout.addWidget(self.btn_mobil_yayin)
            
            self.lbl_yayin_durumu = QLabel("Yayın Durumu: ÇEVRİMDIŞI")
            self.lbl_yayin_durumu.setAlignment(Qt.AlignCenter)
            self.lbl_yayin_durumu.setStyleSheet("background-color: #1E1E1E; color: #E74C3C; border-radius: 8px; padding: 6px; font-weight: bold;")
            layout.addWidget(self.lbl_yayin_durumu)

        def mobil_yayini_baslat(self):
            def run_streamlit():
                current_script = os.path.abspath(sys.argv[0])
                command = f'streamlit run "{current_script}" --server.port 8501 --server.address 0.0.0.0 --server.headless true'
                subprocess.Popen(command, shell=True)
                
            threading.Thread(target=run_streamlit, daemon=True).start()
            self.lbl_yayin_durumu.setText("🟢 YAYIN AKTİF - Telefonunuzdan Bağlanabilirsiniz")
            self.lbl_yayin_durumu.setStyleSheet("background-color: #1E1E1E; color: #2ECC71; border-radius: 8px; padding: 6px; font-weight: bold;")
            self.btn_mobil_yayin.setEnabled(False)
            self.btn_mobil_yayin.setText("⚡ MOBİL RADAR ARKA PLANDA ÇALIŞIYOR")

        def hisse_analiz_et(self):
            hisse_kodu = self.hisse_input.text().upper().strip()
            if not hisse_kodu: return
            
            self.lbl_hisse_adi.setText("Analiz Ediliyor...")
            QApplication.processEvents()
            
            sorgu_kodu = hisse_kodu if hisse_kodu.endswith(".IS") else hisse_kodu + ".IS"
            
            try:
                df = yf.download(sorgu_kodu, period="60d", interval="1d", progress=False)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                    
                if df is None or df.empty or len(df) < 5:
                    self.lbl_hisse_adi.setText("Hata")
                    self.lbl_detay.setText("Veri bulunamadı veya sembol kaldırılmış (Delisted).")
                    return

                kapanis = df['Close'].squeeze()
                son_fiyat = kapanis.iloc[-1]
                
                hedef_fiyat, tahmin_serisi = mobil_tahmin_motoru(df)
                potansiyel_getiri = ((hedef_fiyat - son_fiyat) / son_fiyat) * 100 if son_fiyat > 0 else 0
                
                son_rsi = ta.momentum.rsi(kapanis, window=14).iloc[-1]
                macd_obj = ta.trend.MACD(kapanis)
                macd_cizgisi = macd_obj.macd().iloc[-1]
                macd_sinyal = macd_obj.macd_signal().iloc[-1]
                
                if (son_rsi < 42 and macd_cizgisi > macd_sinyal) or (son_rsi < 30):
                    genel_durum = "AL"
                    self.lbl_sinyal.setStyleSheet("background-color: #2ECC71; color: #121212; border-radius: 8px; padding: 6px; font-weight: bold;")
                elif (son_rsi > 70) or (macd_cizgisi < macd_sinyal):
                    genel_durum = "SAT"
                    self.lbl_sinyal.setStyleSheet("background-color: #C0392B; color: white; border-radius: 8px; padding: 6px; font-weight: bold;")
                else:
                    genel_durum = "TUT"
                    self.lbl_sinyal.setStyleSheet("background-color: #2D2D2D; color: #AAA; border-radius: 8px; padding: 6px; font-weight: bold;")

                self.lbl_sinyal.setText(genel_durum)
                detay_metni = f"RSI: {son_rsi:.2f} | MACD Durumu: {genel_durum}"
                tahmin_mesaji = f"\n\n🚀 YZ 5 Günlük Tahmin: {hedef_fiyat:.2f} TL (Potansiyel: %{potansiyel_getiri:+.2f})"
                detay_metni += tahmin_mesaji
                
                self.lbl_hisse_adi.setText(hisse_kodu)
                self.lbl_fiyat.setText(f"{son_fiyat:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", "."))
                self.lbl_detay.setText(detay_metni)
                
                self.grafik_ciz(df, hisse_kodu, tahmin_serisi)
                
            except Exception as e:
                self.lbl_hisse_adi.setText("Hata")
                self.lbl_detay.setText(f"Analiz yapılamadı: {str(e)}")

        def grafik_ciz(self, df, hisse_adi, tahmin_serisi):
            if df is None or df.empty: return
            self.grafik_fig.clear()
            ax = self.grafik_fig.add_subplot(111)
            ax.set_facecolor('#1E1E1E')
            
            kapanislar = df['Close'].squeeze().tail(30)
            gunler = np.arange(len(kapanislar))
            
            ax.plot(gunler, kapanislar.values, color='#00F0FF', linewidth=2, label="Gerçek")
            ax.fill_between(gunler, kapanislar.values, min(kapanislar.values)*0.99, color='#00F0FF', alpha=0.08)
            
            tahmin_gunler = np.arange(len(kapanislar)-1, len(kapanislar) + 4)
            tahmin_degerleri = np.concatenate(([kapanislar.iloc[-1]], tahmin_serisi))
            ax.plot(tahmin_gunler, tahmin_degerleri, color='#FF00FF', linestyle='--', linewidth=2, label="YZ Tahmin")
            
            ax.tick_params(colors='white', labelsize=8)
            ax.grid(True, color='#2D2D2D', linestyle='--')
            ax.legend(loc='upper left', fontsize=8, facecolor='#1E1E1E', labelcolor='white')
            
            for spine in ax.spines.values(): spine.set_visible(False)
            self.grafik_fig.tight_layout()
            self.grafik_ekrani.draw()

        def mega_taramayi_baslat(self):
            self.tarayici_liste.clear()
            self.btn_taramayi_baslat.setEnabled(False)
            toplam = len(TUM_BIST)
            bulunan = 0
            
            for i, h in enumerate(TUM_BIST, 1):
                self.lbl_radar_durum.setText(f"Canlı Liste Taranıyor: {h} ({i}/{toplam})")
                QApplication.processEvents()
                
                try:
                    df = yf.download(h + ".IS", period="40d", interval="1d", progress=False)
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                    
                    if df is None or df.empty or len(df) < 15: 
                        continue
                    
                    kapanis = df['Close'].squeeze()
                    son_rsi = ta.momentum.rsi(kapanis, window=14).iloc[-1]
                    macd_obj = ta.trend.MACD(kapanis)
                    macd_cizgisi = macd_obj.macd().iloc[-1]
                    macd_sinyal = macd_obj.macd_signal().iloc[-1]
                    
                    if (son_rsi < 40 and macd_cizgisi > macd_sinyal) or (son_rsi < 30):
                        fiyat = kapanis.iloc[-1]
                        metin = f"🟢 {h:<6} Fiyat: {fiyat:>6,.2f} TL  (RSI: {son_rsi:.1f}) -> AL SİNYALİ"
                        metin = metin.replace(",", "X").replace(".", ",").replace("X", ".")
                        
                        item = QListWidgetItem(metin)
                        item.setForeground(QColor("#2ECC71"))
                        item.setFont(QFont("Courier New", 9, QFont.Bold))
                        self.tarayici_liste.addItem(item)
                        bulunan += 1
                        self.tarayici_liste.scrollToBottom()
                except:
                    continue
                    
            self.btn_taramayi_baslat.setEnabled(True)
            self.lbl_radar_durum.setText(f"Tarama bitti. Canlı borsada {bulunan} adet fırsat yakalandı.")

        def watchlist_yukle(self):
            self.liste_widget.clear()
            hisseler = self.db.listeyi_getir()
            if not hisseler:
                self.liste_widget.addItem("Listeniz boş. Önce hisse ekleyin.")
                self.lbl_toplam_durum.setText("Portföy boş.")
                return

            toplam_maliyet_hacmi = 0.0
            toplam_guncel_hacim = 0.0

            for h, maliyet, adet in hisseler:
                sorgu_kodu = h if h.endswith(".IS") else h + ".IS"
                try:
                    df = yf.download(sorgu_kodu, period="2d", interval="1d", progress=False)
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                    
                    if df is None or df.empty or len(df) == 0:
                        metin, renk = f"{h:<6} Veri Yok/Delisted", QColor("#FF9800")
                        item = QListWidgetItem(metin)
                        item.setForeground(renk)
                        item.setFont(QFont("Courier New", 8, QFont.Bold))
                        self.liste_widget.addItem(item)
                        continue
                        
                    bugun_fiyat = df['Close'].squeeze().iloc[-1]
                    if maliyet > 0:
                        degisim = ((bugun_fiyat - maliyet) / maliyet) * 100
                        toplam_maliyet_hacmi += (maliyet * adet)
                        toplam_guncel_hacim += (bugun_fiyat * adet)
                        maliyet_metni = f"M:{maliyet:,.1f}"
                    else:
                        dun_fiyat = df['Close'].squeeze().iloc[-2] if len(df) >= 2 else bugun_fiyat
                        degisim = ((bugun_fiyat - dun_fiyat) / dun_fiyat) * 100
                        maliyet_metni = "Takip"

                    if maliyet > 0 and degisim <= -5.0: status, renk = "!! STOP !!", QColor("#E74C3C")
                    elif maliyet > 0 and degisim <= -3.0: status, renk = "STP.UYARI", QColor("#E67E22")
                    elif maliyet > 0 and degisim >= 10.0: status, renk = "KÂR AL", QColor("#2ECC71")
                    elif degisim > 0: status, renk = "YÜKSELİŞ", QColor("#27AE60")
                    else: status, renk = "DÜŞÜŞ", QColor("#C0392B")

                    metin = f"{h:<6} F:{bugun_fiyat:>6,.2f} {maliyet_metni} %{degisim:>+5.1f} {status}"
                except: 
                    metin, renk = f"{h:<6} Bağlantı Yok", QColor("#FF9800")

                metin = metin.replace(",", "X").replace(".", ",").replace("X", ".")
                item = QListWidgetItem(metin)
                item.setForeground(renk)
                item.setFont(QFont("Courier New", 8, QFont.Bold))
                self.liste_widget.addItem(item)

            if toplam_maliyet_hacmi > 0:
                toplam_kar_zarar_yuzde = ((toplam_guncel_hacim - toplam_maliyet_hacmi) / toplam_maliyet_hacmi) * 100
                kasa_metni = f"Kasa: {toplam_maliyet_hacmi:,.2f} TL -> Net Durum: %{toplam_kar_zarar_yuzde:+,.2f}"
                kasa_metni = kasa_metni.replace(",", "X").replace(".", ",").replace("X", ".")
                self.lbl_toplam_durum.setText(kasa_metni)
            else:
                self.lbl_toplam_durum.setText("Maliyet girilmemiş takip hisseleri.")

        def canli_fiyat_guncelle(self):
            if self.tabs.currentIndex() == 0:
                self.watchlist_yukle()
                return
            if self.tabs.currentIndex() == 1:
                hisse_kodu = self.hisse_input.text().upper().strip()
                if not hisse_kodu: return
                sorgu_kodu = hisse_kodu if hisse_kodu.endswith(".IS") else hisse_kodu + ".IS"
                try:
                    QApplication.processEvents()
                    ticker = yf.Ticker(sorgu_kodu)
                    hist = ticker.history(period="1d")
                    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.droplevel(1)
                    
                    if hist is not None and not hist.empty and len(hist) > 0:
                        yeni_fiyat = hist['Close'].squeeze().iloc[-1]
                        self.lbl_fiyat.setText(f"{yeni_fiyat:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", "."))
                except:
                    pass

        def maliyet_guncelle_penceresi(self):
            secili_item = self.liste_widget.currentItem()
            if not secili_item: return
            kod = secili_item.text().split()[0]
            if "Veri" in kod or "Listeniz" in kod: return
            maliyet, ok1 = QInputDialog.getDouble(self, "Maliyet Girişi", f"{kod} Alış Fiyatı:", min=0.0, decimals=2)
            if ok1:
                adet, ok2 = QInputDialog.getInt(self, "Adet Girişi", f"Kaç adet {kod} aldınız?:", min=1)
                if ok2:
                    self.db.hisse_ekle(kod, maliyet, adet)
                    self.watchlist_yukle()

        def sekme_degisti(self, index):
            if index == 0: self.watchlist_yukle()

        def listeye_hisse_ekle(self):
            kod = self.hisse_input.text().upper().strip()
            if not kod: return
            if self.db.hisse_ekle(kod, 0.0, 0):
                QMessageBox.information(self, "Başarılı", f"{kod} listenize eklendi.")

        def listeden_hisse_sil(self):
            secili_item = self.liste_widget.currentItem()
            if not secili_item: return
            kod = secili_item.text().split()[0]
            self.db.hisse_sil(kod)
            self.watchlist_yukle()

        def listeden_analize_gonder(self, item):
            kod = item.text().split()[0]
            if "Veri" in kod or "Listeniz" in kod: return
            self.hisse_input.setText(kod)
            self.tabs.setCurrentIndex(1)
            self.hisse_analiz_et()

        def tarayicidan_analize_gonder(self, item):
            if "fırsat" in item.text(): return
            kod = item.text().split()[1]
            self.hisse_input.setText(kod)
            self.tabs.setCurrentIndex(1)
            self.hisse_analiz_et()

    # Masaüstü tetikleyici
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    pencere = BorsaMobilUygulama() 
    pencere.show()
    sys.exit(app.exec_())

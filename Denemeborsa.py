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

# ==============================================================================
# BAHSETTİĞİMİZ AYAR TAM OLARAK BURAYA GELECEK:
# ==============================================================================
import matplotlib
matplotlib.use('Agg') # Matplotlib'in arkada harici pencere açmasını engeller
logging.getLogger('matplotlib').setLevel(logging.ERROR) # Gereksiz logları kapatır
# ==============================================================================

# STANDART KÜTÜPHANELER
import sqlite3
import subprocess
import threading
import socket

# VERİ ANALİZİ VE GRAFİK KÜTÜPHANELERİ
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt

IS_STREAMLIT = "streamlit" in sys.modules

# 1. ÖNCE SINIFA ERİŞEBİLMEK İÇİN VERİTABANI SINIFINI TANIMLA
class Veritabani:
    def __init__(self):
        self.baglanti = sqlite3.connect("takip_listesi.db")
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

# 2. SONRA STREAMLIT KONTROLÜNÜ YAP
if IS_STREAMLIT:   
    import streamlit as st
    st.set_page_config(page_title="Mobil Borsa", layout="centered")
    
    # Başlık ve Veri çekme
    st.title("📱 Mobil Borsa Paneli")
    db = Veritabani()
    hisseler = db.listeyi_getir()
    
    # --- GEÇİCİ EKLEME FORMU ---
    with st.expander("➕ Yeni Hisse Ekle"):
        yeni_hisse = st.text_input("Hisse Kodu (örn: ASELS)").upper()
        maliyet = st.number_input("Maliyet", value=0.0)
        adet = st.number_input("Adet", value=0)
        if st.button("Kaydet"):
            if yeni_hisse:
                db.hisse_ekle(yeni_hisse, maliyet, adet)
                st.success(f"{yeni_hisse} eklendi!")
                st.rerun()
    
    # 📋 TAKİP LİSTESİ BÖLÜMÜ
    st.subheader("📋 Takip Listesi")
    
    if not hisseler:
        st.warning("Henüz takip listesinde hisse yok.")
    else:
        # Hisseleri şık bir tablo olarak göster
        for h, maliyet, adet in hisseler:
            with st.container(border=True):
                col1, col2 = st.columns(2)
                col1.metric("Hisse", h)
                col2.write(f"Maliyet: **{maliyet} TL**")
                st.write(f"Adet: {adet}")

    # Sayfayı yenileme butonu
    if st.button("🔄 Verileri Yenile"):
        st.rerun()
# 3. MASAÜSTÜ KODLARININ GERİ KALANI BURAYA GELECEK...

# --- AŞAĞISI SADECE MASAÜSTÜ İÇİN ---

# ... (Diğer tüm PyQt importların ve kodların burada kalabilir)

df = pd.read_csv("bist_hisseler.csv")

TUM_BIST = df["kod"].tolist()


# --- EKSİKSİZ TÜM BIST AKTİF HİSSE LİSTESİ (538 ADET) ---
TUM_BISTtt = [
    "A1CAP", "ACSEL", "ADEL", "ADESE", "AEFES", "AFYON", "AGESA", "AGHOL", "AGROT", "AHGAZ", 
    "AKBNK", "AKCNS", "AKENR", "AKFGY", "AKFYE", "AKGRT", "AKMGY", "AKSA", "AKSEN", "ALARK", 
    "ALBRK", "ALCTL", "ALGYO", "ALKA", "ALKIM", "ALMAD", "ALTNY", "ALVES", "ANELE", "ANGEN", 
    "ANHYT", "ANSGR", "ARCLK", "ARDYZ", "ARENA", "ARSAN", "ARTMS", "ARZUM", "ASELS", "ASGYO", 
    "ASTOR", "ASUZU", "ATAGY", "ATAKP", "ATATP", "ATEKS", "ATLAS", "ATSYH", "AVGYO", "AVHOL", 
    "AVOD", "AVTUR", "AYDEM", "AYEN", "AYES", "AYGAZ", "AZTEK", "BAGFS", "BAKAB", "BALAT", 
    "BANVT", "BARMA", "BASGZ", "BAYRK", "BERA", "BEYAZ", "BFREN", "BIENP", "BIGCH", "BIMAS", 
    "BIOEN", "BIZIM", "BJKAS", "BLCYT", "BMTKS", "BNASL", "BOBET", "BORLS", "BORSK", "BOSSA", 
    "BRISA", "BRKO", "BRKSN", "BRKVY", "BRLSM", "BRMEN", "BRSAN", "BRYAT", "BSOKE", "BTCIM", 
    "BUCIM", "BURCE", "BURVA", "BVSAN", "BYDNR", "CANTE", "CARYE", "CCOLA", "CELHA", "CEMAS", 
    "CEMTS", "CIMSA", "CLEBI", "CONSE", "COSMO", "CRDFA", "CUSAN", "CVKMD", "CWENE", "DAGHL", 
    "DAGI", "DAPGM", "DARDL", "DGATE", "DGGYO", "DGNMO", "DIRIT", "DITAS", "DMSAS", "DOAS", 
    "DOCO", "DOGUB", "DOHOL", "DOKTA", "DURDO", "DYOBY", "DZGYO", "EBEBK", "ECILC", "ECZYT", 
    "EDATA", "EDIP", "EGEEN", "EGGUB", "EGPRO", "EGSER", "EKGYO", "EKIZ", "EKSUN", "ELITE", 
    "EMKEL", "ENJSA", "ENKAI", "EPLAS", "ERBOS", "EREGL", "ERSU", "ESCOM", "ESEN", "ETILR", 
    "EUPWR", "EUREN", "EYGYO", "FADE", "FENER", "FLAP", "FMIZP", "FONET", "FORMT", "FRIGO", 
    "FROTO", "FZLGY", "GARAN", "GARFA", "GEDIK", "GEDZA", "GENIL", "GENTS", "GEREL", "GESAN", 
    "GIPTA", "GLBMD", "GLCVY", "GLRYH", "GLYHO", "GOKNR", "GOLTS", "GOODY", "GOZDE", "GRNYO", 
    "GSDDE", "GSDHO", "GSRAY", "GUBRF", "GWIND", "GZNMI", "HALKB", "HATEK", "HEDEF", "HEKTS", 
    "HKTM", "HLGYO", "HRZFT", "HTTBT", "HUBVC", "HUNER", "HURGZ", "ICBCT", "IDEAS", "IDGYO", 
    "IHEVA", "IHGZT", "IHLAS", "IHLGM", "IHYAY", "IMASM", "INDES", "INFO", "INGRM", "INTEM", 
    "INVEO", "IPEKE", "ISATR", "ISBTR", "ISCTR", "ISFIN", "ISGSY", "ISGYO", "ISKPL", "ISMEN", 
    "ISSEN", "ISYAT", "IZENR", "IZFAS", "IZINV", "IZMDC", "JANTS", "KAPLM", "KAREL", "KARSN", 
    "KARTN", "KARYE", "KATMR", "KAYSE", "KCAER", "KCHOL", "KENT", "KERVT", "KFEIN", "KGYO", 
    "KIMMR", "KLRGY", "KMPUR", "KNFRT", "KOBIL", "KOCAER", "KOCMT", "KONTR", "KONYA", "KORDS", 
    "KOZAA", "KOZAL", "KRDMA", "KRDMB", "KRDMD", "KRGYO", "KRONT", "KRPLS", "KRSTL", "KRTEK", 
    "KSTUR", "KTSKR", "KUTPO", "KUVVA", "KUYAS", "KZBGY", "KZGYO", "LIDER", "LIDFA", "LINK", 
    "LKMNH", "LOGAS", "LOGO", "LUKSK", "MAALT", "MACKO", "MAGEN", "MAKIM", "MAKTK", "MANAS", 
    "MARKA", "MARTI", "MAVI", "MEDTR", "MEGAP", "MEPET", "MERCN", "MERKO", "METRO", "METUR", 
    "MHRGY", "MIATK", "MGROS", "MIPAZ", "MMCAS", "MNDRS", "MNDTR", "MOBTL", "MOGAN", "MPARK", 
    "MRGYO", "MRSHL", "MSGYO", "MTRKS", "MTRYO", "MZHLD", "NATEN", "NETAS", "NIBAS", "NTGAZ", 
    "NTHOL", "NUGYO", "NUHCM", "OBASE", "ODAS", "OFSYM", "ONCSM", "ORCAY", "ORGE", "OTKAR", 
    "OYAKC", "OYAYO", "OYLUM", "OYYAT", "OZGYO", "OZKGY", "OZRDN", "OZSUB", "PAGYO", "PAMEL", 
    "PAPIL", "PARSN", "PASEU", "PATEK", "PCILT", "PEGYO", "PEKGY", "PENGD", "PENTA", "PETKM", 
    "PETUN", "PGSUS", "PINSU", "PKART", "PKENT", "PLTUR", "PNLSN", "PNSUT", "POLHO", "POLTK", 
    "PRKAB", "PRKME", "PRZMA", "PSGYO", "QNBFB", "QNBFL", "QUAGR", "RALYH", "RAYYS", "REEDR", 
    "RNPOL", "RODRG", "RTALB", "RUBNS", "SAHOL", "SAMAT", "SANEL", "SANFO", "SANKO", "SARKY", 
    "SASA", "SAYAS", "SDTTR", "SEKFK", "SEKUR", "SELEC", "SELGD", "SELVA", "SEYKM", "SILVR", 
    "SISE", "SKBNK", "SKTAS", "SMART", "SMRTG", "SNAYS", "SNICA", "SNKPA", "SOKM", "SONME", 
    "SRVGY", "SUMAS", "SUNTK", "SURGY", "SUWEN", "TABGD", "TAPDI", "TARKM", "TATGD", "TAVHL", 
    "TCELL", "TDGYO", "TEKTU", "TERA", "TETMT", "TGSAS", "THYAO", "TKFEN", "TKNSA", "TLMAN", 
    "TMPOL", "TMSN", "TOASO", "TRCAS", "TRGYO", "TRILC", "TSGYO", "TSKB", "TSPOR", "TTKOM", 
    "TTRAK", "TUCLK", "TUKAS", "TUPRS", "TUREX", "TURGG", "TURSG", "UFUK", "ULAS", "ULKER", 
    "ULUFA", "ULUSE", "ULUUN", "UMPAS", "USAK", "VAKBN", "VAKFN", "VAKKO", "VANGD", "VBTYZ", 
    "VERTU", "VERUS", "VESBE", "VESTL", "VKFYO", "VKGYO", "VKING", "YAPRK", "YATAS", "YAYLA", 
    "YBTAS", "YEOTK", "YESIL", "YGGYO", "YGYO", "YKBNK", "YKSLN", "YONGA", "YUNSA", "YYAPI", 
    "YYLGD", "ZEDUR", "ZOREN", "ZRGYO"
]

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QPushButton, QLineEdit, QLabel, 
                                 QTabWidget,
                                 QListWidget, QListWidgetItem, QStatusBar, QMessageBox,
                                 QFrame, QGraphicsDropShadowEffect, QInputDialog)


from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtGui import QFont, QColor


import yfinance as yf
import pandas as pd
import ta
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.linear_model import HuberRegressor # Aykırı değerlere (ani düşüş/yükseliş) karşı daha dayanıklı bir model




# --- ANA UYGULAMA PENCERESİ ---
# --- ANA UYGULAMA PENCERESİ ---
class BorsaMobilUygulama(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db = Veritabani()
        self.initUI()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.canli_fiyat_guncelle)
        self.timer.start(10000) # Her 10 saniyede bir günceller
            
    # FONKSİYONU BURAYA, initUI'IN DIŞINA ALIYORUZ
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
        
        # ... (Stil ve Tab kısımları aynı kalacak) ...
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
        
        self.grafik_fig = Figure(figsize=(4, 2), dpi=100, facecolor='#121212')
        self.grafik_ekrani = FigureCanvas(self.grafik_fig)
        
        self.kur_watchlist_sekmesi()
        self.kur_analiz_sekmesi()
        self.kur_tarayici_sekmesi()
        
        self.tabs.currentChanged.connect(self.sekme_degisti)

        # IP Bilgisini en alta ekle
        self.setStatusBar(QStatusBar()) # Status bar'ı tanımla
        port = 8501 # Değiştirdiğimiz port
        self.lbl_ip_bilgi = QLabel(f"Bağlantı: http://{self.yerel_ip_bul()}:{port}")
       
        self.lbl_ip_bilgi.setStyleSheet("color: #00F0FF; font-size: 10px; font-weight: bold;")
        self.statusBar().addWidget(self.lbl_ip_bilgi)

    # --- 1. SEKME: PORTFÖY VE KASA DURUMU ---
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

    # --- 2. SEKME: PANEL KART ANALİZİ ---
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
        layout.addWidget(self.grafik_ekrani) # Neon Çizgi Grafiği Alt Kısma Bağladık

    # --- 3. SEKME: MEGA RADAR & MİLLİ MOBİL YAYIN ŞALTERİ ---
    def kur_tarayici_sekmesi(self):
        layout = QVBoxLayout(self.tab_tarayici)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)
        
        info_lbl = QLabel("📱 iPhone 13 Canlı Yayın Merkezi")
        info_lbl.setFont(QFont("Arial", 13, QFont.Bold))
        info_lbl.setAlignment(Qt.AlignCenter)
        info_lbl.setStyleSheet("color: #00F0FF;")
        layout.addWidget(info_lbl)
        
        # --- LOKAL MASAÜSTÜ RADARI TETİKLEME BUTONU VE LİSTESİ ---
        self.btn_taramayi_baslat = QPushButton("🔍 MASAÜSTÜ RADAR TARAMASI BAŞLAT")
        self.btn_taramayi_baslat.setFont(QFont("Arial", 9, QFont.Bold))
        self.btn_taramayi_baslat.setStyleSheet("QPushButton { background-color: #2D2D2D; color: white; border-radius: 10px; padding: 10px; border: 1px solid #444; }")
        self.btn_taramayi_baslat.clicked.connect(self.mega_taramayi_baslat)
        layout.addWidget(self.btn_taramayi_baslat)

        self.lbl_radar_durum = QLabel("Masaüstü radarı hazır.")
        self.lbl_radar_durum.setStyleSheet("color: #8A8A8A; font-size: 10px;")
        self.lbl_radar_durum.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_radar_durum)

        self.tarayici_liste = QListWidget()
        self.tarayici_liste.setStyleSheet("QListWidget { background-color: #1E1E1E; border: 1px solid #2D2D2D; border-radius: 10px; color: white; }")
        self.tarayici_liste.itemDoubleClicked.connect(self.tarayicidan_analize_gonder)
        layout.addWidget(self.tarayici_liste)
        
        # --- CEBE GÖNDEREN MOBİL YAYIN BUTONU ---
        self.btn_mobil_yayin = QPushButton("🚀 İPHONE MOBİL YAYINI BAŞLAT")
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

    # --- MOBİL YAYIN TETİKLEYİCİ FONKSİYON (THREADİNG) ---
    def mobil_yayini_baslat(self):
        def run_streamlit():
            # Dosya yolunu senin kaydettiğin tam konuma göre çeker
            # Eski: command = 'streamlit run "D:/Ayhan/Borsa/borsa_mobil.py" --server.port 8501 ...'
# Yeni: Portu 8505 yapalım
     #       command = 'streamlit run "D:/Ayhan/Borsa/BorsaGelismis.py" --server.port 8888 --server.address 0.0.0.0 --server.headless true --server.enableXsrfProtection false'
            command = 'streamlit run "D:\\Borsa\\BorsaGelismis.py" --server.port 8501 --server.address 0.0.0.0 --server.headless true'
            subprocess.Popen(command, shell=True)
            
        threading.Thread(target=run_streamlit, daemon=True).start()
        
        self.lbl_yayin_durumu.setText("🟢 YAYIN AKTİF - iPhone'dan Bağlanabilirsiniz")
        self.lbl_yayin_durumu.setStyleSheet("background-color: #1E1E1E; color: #2ECC71; border-radius: 8px; padding: 6px; font-weight: bold;")
        self.btn_mobil_yayin.setEnabled(False)
        self.btn_mobil_yayin.setText("⚡ MOBİL RADAR ARKA PLANDA ÇALIŞIYOR")

    # --- AL-SAT-TUT HESAPLAMA MOTORU ---
    def hisse_analiz_et(self):
            hisse_kodu = self.hisse_input.text().upper().strip()
            if not hisse_kodu: return
            
            self.lbl_hisse_adi.setText("Analiz Ediliyor...")
            QApplication.processEvents()
            
            sorgu_kodu = hisse_kodu if hisse_kodu.endswith(".IS") else hisse_kodu + ".IS"
            
            try:
                df = yf.download(sorgu_kodu, period="60d", interval="1d", progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                    
                if df.empty:
                    self.lbl_hisse_adi.setText("Hata")
                    return

                kapanis = df['Close'].squeeze()
                son_fiyat = kapanis.iloc[-1]
                
                # --- YZ TAHMİN MOTORUNU ÇALIŞTIR ---
                hedef_fiyat, tahmin_serisi = self.tahmin_et_ve_hedef_belirle(df)
                potansiyel_getiri = ((hedef_fiyat - son_fiyat) / son_fiyat) * 100
                
                # --- İNDİKATÖRLER ---
                son_rsi = ta.momentum.rsi(kapanis, window=14).iloc[-1]
                macd_obj = ta.trend.MACD(kapanis)
                macd_cizgisi = macd_obj.macd().iloc[-1]
                macd_sinyal = macd_obj.macd_signal().iloc[-1]
                
                # --- SİNYAL MANTIĞI ---
                if (son_rsi < 42 and macd_cizgisi > macd_sinyal) or (son_rsi < 30):
                    genel_durum = "AL"
                elif (son_rsi > 70) or (macd_cizgisi < macd_sinyal):
                    genel_durum = "SAT"
                else:
                    genel_durum = "TUT"

                # --- EKRAN ÇIKTISI OLUŞTURMA ---
                detay_metni = f"RSI: {son_rsi:.2f} | MACD Durumu: {genel_durum}"
                
                # YZ Tahmin Mesajı
                tahmin_mesaji = f"\n\n🚀 YZ 5 Günlük Tahmin: {hedef_fiyat:.2f} TL (Potansiyel: %{potansiyel_getiri:+.2f})"
                detay_metni += tahmin_mesaji
                
                # UI Güncelleme
                self.lbl_hisse_adi.setText(hisse_kodu)
                self.lbl_fiyat.setText(f"{son_fiyat:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", "."))
                self.lbl_detay.setText(detay_metni)
                
                # Grafiği Tahmin Serisi ile Çiz
                self.grafik_ciz(df, hisse_kodu, tahmin_serisi)
                
            except Exception as e:
                self.lbl_hisse_adi.setText("Hata")
                self.lbl_detay.setText(f"Analiz yapılamadı: {str(e)}")

    # --- ŞIK NEON ÇİZGİ GRAFİK ---
    def grafik_ciz(self, df, hisse_adi, tahmin_serisi):
            self.grafik_fig.clear()
            ax = self.grafik_fig.add_subplot(111)
            ax.set_facecolor('#1E1E1E')
            
            kapanislar = df['Close'].squeeze().tail(30)
            gunler = np.arange(len(kapanislar))
            
            # Gerçek Fiyat Çizgisi
            ax.plot(gunler, kapanislar.values, color='#00F0FF', linewidth=2, label="Gerçek")
            ax.fill_between(gunler, kapanislar.values, min(kapanislar.values)*0.99, color='#00F0FF', alpha=0.08)
            
            # YZ Tahmin Çizgisi (Mor - Kesikli)
            tahmin_gunler = np.arange(len(kapanislar)-1, len(kapanislar) + 4)
            tahmin_degerleri = np.concatenate(([kapanislar.iloc[-1]], tahmin_serisi))
            ax.plot(tahmin_gunler, tahmin_degerleri, color='#FF00FF', linestyle='--', linewidth=2, label="YZ Tahmin")
            
            # Stil Ayarları
            ax.tick_params(colors='white', labelsize=8)
            ax.grid(True, color='#2D2D2D', linestyle='--')
            ax.legend(loc='upper left', fontsize=8, facecolor='#1E1E1E', labelcolor='white')
            
            for spine in ax.spines.values():
                spine.set_visible(False)
                
            self.grafik_fig.tight_layout()
            self.grafik_ekrani.draw()
    # --- GERÇEK TÜM PAZAR RADAR MOTORU ---
    def mega_taramayi_baslat(self):
        self.tarayici_liste.clear()
        self.btn_taramayi_baslat.setEnabled(False)
        toplam = len(TUM_BIST)
        bulunan = 0
        
        for i, h in enumerate(TUM_BIST, 1):
            self.lbl_radar_durum.setText(f"Borsa Tamamı Taranıyor: {h} ({i}/{toplam})")
            QApplication.processEvents()
            
            try:
                df = yf.download(h + ".IS", period="40d", interval="1d", progress=False)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                
                if df.empty: continue
                
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
        self.lbl_radar_durum.setText(f"Tarama bitti. Tüm borsada {bulunan} adet fırsat yakalandı.")

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
                
                if not df.empty:
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
                else: metin, renk = f"{h:<6} Veri Hatası", QColor("#FF9800")
            except: metin, renk = f"{h:<6} Bağlantı Yok", QColor("#FF9800")

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
        
    
    def tahmin_et_ve_hedef_belirle(self, df):
            # 1. Veriyi hazırla
            data = df.tail(60).copy()
            data['gun'] = range(len(data))
            
            # Sütun ismini koruyarak DataFrame olarak bırakıyoruz
            X_train = data[['gun']] 
            y_train = data['Close']
            
            # 2. Modeli eğit
            model = HuberRegressor(max_iter=1000)
            model.fit(X_train, y_train)
            
            # 3. Gelecek günleri DataFrame olarak hazırla
            son_gun_index = data['gun'].iloc[-1]
            gelecek_gunler = pd.DataFrame({'gun': range(son_gun_index + 1, son_gun_index + 6)})
            
            # Tahminler
            tahmin_fiyat = model.predict(gelecek_gunler.tail(1))[0]
            tahmin_serisi = model.predict(gelecek_gunler)
            
            return tahmin_fiyat, tahmin_serisi

    def canli_fiyat_guncelle(self):
        """
        QTimer tarafından çağrılan, arayüzü dondurmayan 
        ve MultiIndex hatalarına karşı korumalı canlı fiyat motoru.
        """
        # Mantık Optimizasyonu: Eğer kullanıcı portföy sekmesindeyse, tüm listeyi sessizce yenile
        if self.tabs.currentIndex() == 0:
            self.watchlist_yukle()
            return

        # Eğer analiz sekmesindeyse, ekrandaki hissenin fiyatını güncelle
        if self.tabs.currentIndex() == 1:
            hisse_kodu = self.hisse_input.text().upper().strip()
            if not hisse_kodu: 
                return
            
            sorgu_kodu = hisse_kodu if hisse_kodu.endswith(".IS") else hisse_kodu + ".IS"
            
            try:
                # Arayüze "istek gönderiliyor" nefesi aldır
                QApplication.processEvents()
                
                ticker = yf.Ticker(sorgu_kodu)
                hist = ticker.history(period="1d")
                
                # MultiIndex sütun yapısı varsa temizle
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.droplevel(1)
                    
                if not hist.empty:
                    yeni_fiyat = hist['Close'].squeeze().iloc[-1]
                    # Türk Lirası formatlama (Binlik ayırıcı nokta, ondalık virgül)
                    formatli_fiyat = f"{yeni_fiyat:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    self.lbl_fiyat.setText(formatli_fiyat)
            except Exception as e:
                # Arka plan güncelleme hatası loglanabilir, arayüzü patlatmaması için pass geçiyoruz
                print(f"Canlı fiyat güncellenirken hata oluştu: {e}")


# ==============================================================================
# ANA ÇALIŞTIRMA KONTROLÜ
# ==============================================================================
if __name__ == "__main__":
    # Eğer bu script Streamlit (Mobil) tarafından çağrılmadıysa PyQt5'i başlat
    if not IS_STREAMLIT:
        app = QApplication(sys.argv)
        
        # Koyu Tema Arayüz Optimizasyonu (Windows başlık çubuğu uyumu için)
        app.setStyle('Fusion')
        
        pencere = BorsaMobilUygulama()
        pencere.show()
        sys.exit(app.exec_())

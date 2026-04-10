import sys
import os

import PyQt5

# [HATA DÜZELTME] Windows PyQt5 için QPA Eklenti Yolunu Klasörden Direkt Çekme
pyqt_plugins = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = pyqt_plugins
os.environ['QT_PLUGIN_PATH'] = pyqt_plugins

import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QTextEdit, QLineEdit, QPushButton, QLabel, QCheckBox)
from PyQt5.QtGui import QFont, QTextCursor, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from deep_translator import GoogleTranslator

# ==============================================================================
# BÖLÜM 1: GÜVENLİK VE BULUT BAĞLANTISI
# ==============================================================================
HF_TOKEN = "hf_YOUR_TOKEN_HERE"
REPO_ID = "OnurDemircioglu/OmniGPT-355M"

# ==============================================================================
# BÖLÜM 2: ARKA PLAN YAPAY ZEKA MOTORU (UI DONMASINI ENGELLER)
# ==============================================================================
class ModelMotoru(QThread):
    sinyal_durum = pyqtSignal(str)
    sinyal_cevap = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.hazir = False
        self.islem_modu = "YUKLE" # 'YUKLE' veya 'CEVAPLA'
        self.kullanici_sorusu = ""
        self.translator_tr_to_en = GoogleTranslator(source='tr', target='en')
        self.translator_en_to_tr = GoogleTranslator(source='en', target='tr')
        self.ceviri_aktif = False

    def run(self):
        if self.islem_modu == "YUKLE":
            try:
                self.sinyal_durum.emit(f"[*] Hugging Face (Kasa) Anahtarı Onaylanıyor...")
                login(token=HF_TOKEN, add_to_git_credential=False)
                
                # Cihaz Tespiti
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.sinyal_durum.emit(f"[*] Ağırlıklar (1.42 GB) Süzülüyor... ({self.device.upper()} Aktif)")
                # Tokenizer'ı standart GPT-2'den çekiyoruz (Kelime haritamız aynı olduğu için daha sağlıklı olur)
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.sinyal_durum.emit(f"[*] OmniGPT-355M Beyni VRAM'e Kuruluyor (GPU!)...")
                # GPU varsa FP16 kullan (Hızlandırır ve RAM kurtarır)
                dt = torch.float16 if self.device == "cuda" else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(REPO_ID, token=HF_TOKEN, torch_dtype=dt).to(self.device)
                self.model.eval() # Eğitimi tamamen kapat.
                
                self.hazir = True
                self.sinyal_durum.emit(f"[🚀] OmniGPT Çevrimiçi! Lokal GPU'da Çalışıyor.")
            except Exception as e:
                self.sinyal_durum.emit(f"[⚠️] HATA OLUŞTU: {e}")

        elif self.islem_modu == "CEVAPLA" and self.hazir:
            try:
                soru = self.kullanici_sorusu
                if self.ceviri_aktif:
                    self.sinyal_durum.emit("Çevriliyor: TR -> EN...")
                    try:
                        soru = self.translator_tr_to_en.translate(soru)
                    except Exception as e:
                        print(f"Çeviri hatası (Input): {e}")

                # Kelimeyi Token'a (Matrislere) dönüştür
                self.sinyal_durum.emit(f"OmniGPT Düşünüyor...")
                inputs = self.tokenizer(soru, return_tensors='pt').to(self.device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                
                # GPU Inference - Hızlı Üretim
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,       # Cümle uzunluğu 
                    temperature=0.7,         # Yaratıcılık oranı (Sıcaklık)
                    top_p=0.9,               # Nükleus örnekleme (Mantıklı kelime seçimi)
                    top_k=50,
                    repetition_penalty=1.2,  # Takılıp kalmasını engelle
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    no_repeat_ngram_size=2
                )
                
                # Çıktıyı Metne Geri Çevir
                uretilen_metin = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Kullanıcının sorduğu kelimeyi baştan kes (Sadece modelin ürettiği kısımları al)
                cevap = uretilen_metin[len(soru):].strip()
                if len(cevap) == 0: cevap = "..." # Boş kalırsa
                
                orjinal_cevap = cevap
                tam_basilacak = orjinal_cevap
                
                if self.ceviri_aktif:
                    self.sinyal_durum.emit("Çevriliyor: EN -> TR...")
                    try:
                        cevap_tr = self.translator_en_to_tr.translate(orjinal_cevap)
                        # HTML ile alt alta iki versiyonu basıyoruz
                        tam_basilacak = f"<span style='color:#00ffcc'>{cevap_tr}</span><br><br><i style='color:#557766'>[ORİJİNAL İNGİLİZCE]: {orjinal_cevap}</i>"
                    except Exception as e:
                        print(f"Çeviri hatası (Output): {e}")

                self.sinyal_cevap.emit(tam_basilacak)
                self.sinyal_durum.emit(f"[🚀] OmniGPT Çevrimiçi! Lokal GPU'da Çalışıyor.")
            except Exception as e:
                self.sinyal_cevap.emit(f"[⚠️] Üretim Hatası: {e}")
                self.sinyal_durum.emit(f"[🚀] OmniGPT Çevrimiçi! Lokal GPU'da Çalışıyor.")

# ==============================================================================
# BÖLÜM 3: PYQT5 HACKER ARAYÜZÜ (GUI)
# ==============================================================================
class OmniGPTEkran(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OmniGPT-355M (Canlı Test İstasyonu)")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #0b0f19; color: #00ffcc;")
        self.arayuz_kur()
        
        # Motoru Başlat
        self.motor = ModelMotoru()
        self.motor.sinyal_durum.connect(self.durum_guncelle)
        self.motor.sinyal_cevap.connect(self.cevap_yansit)
        
        # İlk Yükleme
        self.motor.islem_modu = "YUKLE"
        self.motor.start()

    def arayuz_kur(self):
        merkez_widget = QWidget()
        self.setCentralWidget(merkez_widget)
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # 1. Başlık
        baslik = QLabel(" OMNI-GPT 355M MÜHENDİSLİK KONSOLU")
        baslik.setFont(QFont("Consolas", 18, QFont.Bold))
        baslik.setAlignment(Qt.AlignCenter)
        baslik.setStyleSheet("color: #ffaa00; margin-bottom: 10px;")
        layout.addWidget(baslik)
        
        # 2. Durum Çubuğu
        self.durum_label = QLabel("[*] Sistem başlatılıyor...")
        self.durum_label.setFont(QFont("Consolas", 10))
        self.durum_label.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(self.durum_label)

        # 3. Chat Ekranı
        self.chat_ekrani = QTextEdit()
        self.chat_ekrani.setReadOnly(True)
        self.chat_ekrani.setFont(QFont("Consolas", 12))
        self.chat_ekrani.setStyleSheet("""
            background-color: #121826;
            border: 1px solid #00ffcc;
            border-radius: 5px;
            padding: 10px;
        """)
        layout.addWidget(self.chat_ekrani)

        # 4. Girdi Alanı
        self.girdi = QLineEdit()
        self.girdi.setFont(QFont("Consolas", 12))
        self.girdi.setPlaceholderText("Mesajınızı yazıp 'Enter'a basın...")
        self.girdi.setStyleSheet("""
            background-color: #1a2235;
            color: #ffffff;
            border: 1px solid #ffaa00;
            border-radius: 5px;
            padding: 10px;
        """)
        self.girdi.returnPressed.connect(self.gonderildi)
        layout.addWidget(self.girdi)
        
        # 5. Çeviri Köprüsü Modülü (Toggle)
        self.ceviri_check = QCheckBox("🌐 İki Yönlü İngilizce <-> Türkçe Çeviri Köprüsünü Aktif Et")
        self.ceviri_check.setFont(QFont("Consolas", 10))
        self.ceviri_check.setStyleSheet("color: #00ffcc; margin-top: 5px;")
        self.ceviri_check.setChecked(False)
        self.ceviri_check.stateChanged.connect(self.ceviri_degisti)
        layout.addWidget(self.ceviri_check)

        merkez_widget.setLayout(layout)

    def ceviri_degisti(self, state):
        self.motor.ceviri_aktif = (state == Qt.Checked)

    def durum_guncelle(self, mesaj):
        self.durum_label.setText(mesaj)
        
    def ekrana_yaz(self, gonderen, metin, renk):
        self.chat_ekrani.moveCursor(QTextCursor.End)
        # HTML ile renklendirme
        satir = f'<b style="color:{renk}">[{gonderen}]</b>: <span style="color:#ffffff">{metin}</span><br><br>'
        self.chat_ekrani.insertHtml(satir)
        self.chat_ekrani.moveCursor(QTextCursor.End)

    def gonderildi(self):
        if not self.motor.hazir:
            self.ekrana_yaz("SİSTEM", "Bekleyin... Model henüz yüklenmedi (1.4 GB boyutunda).", "#ff0000")
            return
            
        mesaj = self.girdi.text().strip()
        if not mesaj: return
        
        self.girdi.clear()
        self.ekrana_yaz("SİZ", mesaj, "#ffaa00")
        
        # Modele Emir Ver
        self.motor.kullanici_sorusu = mesaj
        self.motor.islem_modu = "CEVAPLA"
        self.motor.start()

    def cevap_yansit(self, cevap):
        # Arayüze dönen cevabı direkt ekle (HTML formatıyla sarmalanmış olabilir)
        self.chat_ekrani.moveCursor(QTextCursor.End)
        satir = f'<b style="color:#00ffcc">[OMNIGPT-355M]</b>:<br>{cevap}<br><br>'
        self.chat_ekrani.insertHtml(satir)
        self.chat_ekrani.moveCursor(QTextCursor.End)

if __name__ == '__main__':
    uygulama = QApplication(sys.argv)
    pencere = OmniGPTEkran()
    pencere.show()
    sys.exit(uygulama.exec_())

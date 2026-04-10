# OmniGPT-355M: Causal LLM Fine-Tuning & Knowledge Distillation

![Python](https://img.shields.io/badge/Python-3.13-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-GPU_Optimized-ee4c2c.svg) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Integrated-FFD21E.svg) ![PyQt5](https://img.shields.io/badge/PyQt5-GUI-green.svg)

**OmniGPT-355M**, OpenAI'nin GPT-2 Medium (355 Milyon parametre) taban mimarisi üzerinden **LMSYS Chatbot Arena Conversations** veri setini temel alan bir "Teacher-Student Knowledge Distillation" ve Fine-Tuning (İnce Ayar) projesidir.

Bu projedeki temel mühendislik hedefi, devasa büyüklükteki modern dil modellerinin (Örn. GPT-4, Claude 3) sofistike akıl yürütme becerilerini ve diyalog dinamiklerini, oldukça küçük ve optimize edilebilir 355 milyon parametrelik bir "Öğrenci Modele" donanımsal taşmalar yaşamadan transfer etmektir. 

---

## 🛠️ Temel Sistem Mimarisi ve MLOps Altyapısı

### 1. Eğitim (Training) Altyapısı - `train_omnigpt.py`
Bu dosya Kaggle üzerindeki devasa sistemlerde yürütülmek üzere tasarlanmıştır.

#### 🧠 Yapısal ve Matematiksel Parametreler (Model Anatomy)
GPT-2 Medium Mimarisine ait katı çerçeveler (HuggingFace Config) projeye şu şekilde yansıtılmıştır:
* **Max Sequence Length (Bağlam Uzunluğu):** 256
* **Vocabulary Size (Sözlük Boyutu):** 50,257 boyutlu BPE (Byte-Pair Encoding) Mimarisi 
* **Model Sıkleti (Parameters):** ~355M
* **`n_embd` (Embedding Boyutu):** 1024
* **`n_layer` (Transformer Katmanları):** 24
* **`n_head` (Attention Heads):** 16

#### ⚙️ Eğitim Operasyonları ve MLOps
* **Donanım:** Dual NVIDIA Tesla T4 (GPU)
* **Bellek Optimizasyonları (OOM Fixes):** Büyük matris boyutlarının 16GB'lık VRAM'e sığması için standart AdamW yerine bellek tasarrufu sağlayan **Adafactor** kullanılmış ve hesaplama grafiklerine **Gradient Checkpointing** ($use\_reentrant=False$) entegre edilmiştir. Ayrıca FP16 Precision kullanılmıştır.
* **Bulut Kararlılığı (Cloud-Persistent MLOps):** Eğitim sürelerindeki donanımsal kopmalara karşı, Hugging Face Hub üzerinden otomatik `push_to_hub` entegrasyonu yazılmıştır. Model eğitiminin her `save_step` aşamasında ağırlıklar (`safetensors`) HTTP-503 gibi ağ hatalarına karşı *Auto-Retry* mekanizması ile güvenceye alınmıştır.
* **Veri Tüketimi (Recursive Ingestion):** İç içe geçmiş kompleks `JSON` Chatbot Arena diyaloglarını, modelin nedensel yapısına (Causal LM) uygun bir flat-list yapısına çeken recursive (özyinelemeli) özel veri temizleme fonksiyonu geliştirilmiştir.

### 2. İki Yönlü Çeviri & Çıkarım (Inference) Arayüzü - `inference_gui.py`
OmniGPT-355M, saf GPT-2 mimarisine dayandığı için ana akıl yürütme süreçleri İngilizcedir. Masaüstü uygulamasında bu engeli aşmak için **Bilateral Translation Pipeline (İki Yönlü Çeviri Köprüsü)** oluşturulmuştur.

* **Donanım:** Uygulama, kullanıcının yerel CUDA (NVIDIA GPU) birimini algılayarak bulut bağımsız çalışır.
* **Akış Senaryosu:** 
  1. Kullanıcıdan gelen "TR" dizinleri `deep-translator` üzerinden "EN" olarak sisteme sokulur.
  2. Modelin oluşturduğu tahmin ağı (Max Tokens, Temperature, Top-K tabanlı) GPU'da işlenir.
  3. Üretilen (EN) mantıksal döngüler, milisaniye cinsinden çevrilerek kullanıcı arayüzüne çift yönlü kanıtlanabilir formatta sunulur.
* **GUI (Arayüz):** Karanlık tema ekseninde özelleştirilmiş PyQt5 tabanlı lokal grafik arayüz.

---

## 🚀 Kurulum & Çalıştırma (Kullanım Rehberi)

### Gereksinimler
Tüm sistem gereksinimlerini yüklemek için (Virtual Environment tavsiye edilir):
```bash
pip install torch transformers huggingface-hub deep-translator PyQt5
```

### Bağlantı 
Her iki kod dosyasında da yer alan `HF_TOKEN` değerlerine Kendi Hugging Face Write (Yazım) iznine sahip token değerinizi girmelisiniz.

### Eğitim Döngüsünü Başlatmak İçin (Kaggle/Colab)
```bash
python train_omnigpt.py
```

### Arayüzü Başlatmak İçin (Lokal Windows/Linux Ortamı)
```bash
python inference_gui.py
```

---

*Bu proje, Yapay Zeka Kaput Altı matematiğini kavrama ve verimli Donanım Kullanımı (Optimization) hedefleri doğrultusunda Onur Demircioğlu tarafından kurgulanmıştır.*

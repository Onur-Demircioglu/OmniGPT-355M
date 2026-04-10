# ==============================================================================
# BÖLÜM 1: GPU KONTROL VE BAŞLATMA
# ==============================================================================
import os
import json
import gc
import sys
import torch
from huggingface_hub import login

# [GÜVENLİK PROTOKOLÜ] Hugging Face bağlantısı
HF_TOKEN = "hf_YOUR_TOKEN_HERE"
login(token=HF_TOKEN, add_to_git_credential=True)

print("[*] Donanım Taraması Gerçekleştiriliyor...")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[+] ŞAHANE! NVIDIA GPU Bulundu: {gpu_name}")
else:
    print("[-] UYARI: GPU bulunamadı! İşlemci (CPU) ile eğitim asırlar sürebilir.")

print(r"""
=============================================================================
 🗣️ OMNI-GPT (355M) ⚡ SAF GPU GÜCÜ — LLM EĞİTİM MOTORU 🗣️
=============================================================================
 Mimari: GPT-2 Medium (Causal Decoder-Only Transformer)
 Parametre: ~355M (n_embd=1024, n_layer=24, n_head=16)
 Donanım: NVIDIA GPU (T4 / P100) — Sıfır Bekleme, Anında Ateşleme!
=============================================================================
""")


# ==============================================================================
# BÖLÜM 1: KONFİGÜRASYON
# ==============================================================================
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, Adafactor
from torch.utils.data import Dataset

print("[*] GPT-2 BPE Tokenizer yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
MAX_SEQ_LEN = 256  # 512 de denenebilir ama 256 en hızlısı

# ==============================================================================
# BÖLÜM 2: VERİ OKUMA
# ==============================================================================
def icerik_sogurucu(obje):
    metinler = []
    if isinstance(obje, dict):
        for anahtar, deger in obje.items():
            if anahtar == "content" and isinstance(deger, str) and len(deger) > 10:
                metinler.append(deger[:1024])
            else: metinler.extend(icerik_sogurucu(deger))
    elif isinstance(obje, list):
        for eleman in obje: metinler.extend(icerik_sogurucu(eleman))
    return metinler

def veri_yukle():
    bulunan_dosya = None
    for root, _, files in os.walk('/kaggle/input'):
        for f in files:
            if f.endswith('.json'):
                bulunan_dosya = os.path.join(root, f)
                break
    if not bulunan_dosya: return ["Örnek veri setiniz bulunamadı."]
    
    with open(bulunan_dosya, 'r', encoding='utf-8') as f:
        try: data = json.load(f)
        except: 
            f.seek(0)
            data = [json.loads(l) for l in f if l.strip()]
    return icerik_sogurucu(data)

# ==============================================================================
# BÖLÜM 3: DATASET
# ==============================================================================
class GPTDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len, padding='max_length', return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': enc['input_ids'].squeeze(0)
        }

texts = veri_yukle()
dataset = GPTDataset(texts, tokenizer, MAX_SEQ_LEN)
print(f"[+] {len(texts)} diyalog ile eğitim başlıyor.")

# ==============================================================================
# BÖLÜM 4: GPT-2 MEDIUM (355M) GERÇEK MİMARİ
# ==============================================================================
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=MAX_SEQ_LEN,
    n_embd=1024,   # 355M için 1024
    n_layer=24,    # 355M için 24
    n_head=16,     # 355M için 16
    n_inner=4096,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
model = GPT2LMHeadModel(config)
# 16GB VRAM'e 355M'i sığdırmak için Gradient Checkpointing şart!
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# ==============================================================================
# BÖLÜM 5: GPU TRAINER AYARLARI
# ==============================================================================
KAYIT_DIZINI = '/kaggle/working/OmniGPT-355M-GPU'

training_args = TrainingArguments(
    output_dir=KAYIT_DIZINI,
    num_train_epochs=2,             # Zaman kısıtlaması BİTTİ! Standart eğitime dönüyoruz.
    per_device_train_batch_size=2,  # GPU'ya (16GB) sığdırmak için güvenli sınır.
    gradient_accumulation_steps=8,  # Efektif batch 16 (2*8)
    optim="adafactor",              # VRAM tasarrufu için Adafactor kullanmaya devam ediyoruz.
    learning_rate=5e-5,
    fp16=True,                      # [GÜÇ!] NVIDIA GPU'lar için donanımsal hızlandırma.
    
    dataloader_num_workers=2,       # GPU'da çoklu işlemci kullanmak güvenlidir, hızı artırır.
    report_to="none",
    
    # === HUGGING FACE BULUT YEDEKLEME SİSTEMİ ===
    push_to_hub=True,
    hub_model_id="HF_KULLANICI_ADİNİ_BURAYA_YAZ/OmniGPT-355M", # Kendi Kullanıcı adını ekle! (Örn: onurxyz/OmniGPT-355M)
    hub_private_repo=True,          # Sadece sen görebilirsin (Özel Depo)
    hub_strategy="checkpoint",      # Her kayıtta otomatik Upload et
    
    logging_steps=20,
    logging_first_step=True,
    save_steps=300,                 # Her ~30 dakikada bir fırlat
    save_total_limit=1,             # Yerele (Kaggle'a) acımaz, diski doldurmaz.
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# ==============================================================================
# BÖLÜM 6: EĞİTİM
# ==============================================================================
print(f"\n[🚀] NVIDIA GPU GÜCÜ ATEŞLENDİ! (Sıfır Bekleme, Anlık Akış)")

try:
    trainer.train()
    trainer.save_model(KAYIT_DIZINI)
    tokenizer.save_pretrained(KAYIT_DIZINI)
    print(f"\n[🎯] ŞAH MAT! 355M Model Başarıyla Kaydedildi: {KAYIT_DIZINI}")
except (Exception, KeyboardInterrupt) as e:
    print(f"\n[⚠️] EĞİTİM YARIDA KESİLDİ VEYA DURDURULDU: {e}")
    trainer.save_model(KAYIT_DIZINI + "-ACIL_DURUM_KAYDI")
    tokenizer.save_pretrained(KAYIT_DIZINI + "-ACIL_DURUM_KAYDI")
    torch.save(model.state_dict(), "/kaggle/working/omnigpt_355m_weights.pth")
    print("\n[🛡️] ACİL DURUM PROTOKOLÜ: Model o ana kadar öğrendikleriyle MÜHÜRLENDİ!")

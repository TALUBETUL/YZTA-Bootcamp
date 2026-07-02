# Predictive Retention AI 🧠

Telekomünikasyon sektöründe müşteri kaybını (churn) önlemek için geliştirilmiş uçtan uca yapay zeka sistemi.

## 🚀 Özellikler

- **Churn Tahmini** — XGBoost ile yüksek doğruluklu tahmin (Gerçekleşen: %61 F1, %83 ROC-AUC)
- **Açıklanabilirlik (XAI)** — SHAP ile "neden gidiyor?" sorusunun cevabı
- **Kişiselleştirilmiş Mesaj** — Groq AI (llama-3.3-70b) ile geri kazanım e-postası üretimi
- **İnteraktif Dashboard** — Streamlit ile görsel risk analizi paneli

## 📦 Kurulum

```bash
# 1. Gerekli kütüphaneleri yükle
pip install -r requirements.txt

# 2. API anahtarını ayarla
cp .env.example .env
# .env dosyasını açıp GROQ_API_KEY değerini gir
# Ücretsiz API key: https://console.groq.com

# 3. Modeli eğit
python train.py

# 4. Uygulamayı başlat
streamlit run app/streamlit_app.py
```

## 🗂️ Proje Yapısı

```
predictive-retention-ai/
├── data/
│   ├── raw/          # Ham veri (Telco Churn XLS)
│   └── processed/    # Eğitilmiş veri split'leri
├── src/
│   ├── data/         # Veri yükleme ve ön işleme
│   ├── models/       # XGBoost model pipeline
│   ├── xai/          # SHAP açıklanabilirlik
│   └── llm/          # Groq API entegrasyonu
├── app/
│   └── streamlit_app.py  # 3 sayfalı web arayüzü
├── models/           # Eğitilmiş model dosyaları
├── train.py          # Model eğitim script'i
└── requirements.txt
```

## 🛠️ Teknoloji Yığını

| Kategori | Araç |
|----------|------|
| ML Modeli | XGBoost |
| Sınıf Dengesi | SMOTE (imbalanced-learn) |
| XAI | SHAP (TreeExplainer) |
| LLM | Groq API — llama-3.3-70b |
| Arayüz | Streamlit |
| Görselleştirme | Plotly |

## 📊 Veri Seti

IBM Telco Customer Churn Dataset
- **7.043 müşteri**, 21 özellik
- Churn oranı: ~%26

## 🎯 Model Performansı

| Metrik | Hedef | Gerçekleşen (Baseline) |
|--------|-------|------------------------|
| F1 Score | ≥ 0.80 | ~0.61 |
| ROC-AUC | ≥ 0.85 | ~0.83 |

> *Not: Churn sınıfındaki dengesizlik (F1) skorunu daha da artırmak için `python train.py --optimize` komutuyla Optuna optimizasyonunu çalıştırabilirsiniz.*

## 💡 Kullanım

### Model Eğitimi

```bash
# Standart eğitim
python train.py

# Optuna ile hiperparametre optimizasyonu (daha uzun sürer)
python train.py --optimize
```

### Streamlit Arayüzü

Uygulama 3 sayfadan oluşur:

1. **📊 Ana Dashboard** — Tüm müşterilerin risk dağılımı ve yüksek riskli müşteri tablosu
2. **👤 Müşteri Analizi** — Tek müşteri detay analizi + SHAP waterfall grafiği
3. **✉️ Mesaj Üretici** — Groq AI ile kişiselleştirilmiş geri kazanım e-postası

## 🔑 Groq API Kurulumu

1. [console.groq.com](https://console.groq.com) adresine git
2. Ücretsiz hesap aç → API Key oluştur
3. `.env` dosyasına ekle: `GROQ_API_KEY=gsk_...`

# Predictive Retention AI 🧠

Telekomünikasyon sektöründe müşteri kaybını (churn) önlemek için geliştirilmiş uçtan uca yapay zeka sistemi.

## 🚀 Özellikler

- **Churn Tahmini** — XGBoost ile yüksek doğruluklu tahmin (Gerçekleşen: %61 F1, %83 ROC-AUC)
- **Açıklanabilirlik (XAI)** — SHAP ile "neden gidiyor?" sorusunun cevabı
- **Kişiselleştirilmiş Mesaj** — Groq AI (llama-3.3-70b) ile geri kazanım e-postası üretimi
- **İnteraktif Dashboard** — Streamlit ile görsel risk analizi paneli
- **Canlı Tahmin** — CustomerID arama ve yeni müşteri formu
- **Karar Desteği** — Precision-Recall, confusion matrix ve maliyet bazlı risk eşiği
- **Model Karşılaştırması** — Leakage-safe CV ile Logistic Regression, Random Forest ve XGBoost
- **Yönetici Özeti** — Yüksek riskli müşteri grubu için indirilebilir LLM raporu
- **Kalibre Risk Skorları** — Out-of-fold Platt kalibrasyonu ve güvenilirlik raporu
- **Next Best Action** — Teklif maliyeti, müşteri değeri ve beklenen net değer hesabı
- **Model Güveni** — Veri drift, performans ve segment bazlı hata analizi
- **Kampanya Operasyonları** — A/B ataması, ölçülen uplift, insan onayı ve audit geçmişi
- **CRM Aktarımı** — Yalnızca onaylı kayıtlar için CSV veya HTTPS webhook

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

Veri seti hem proje içindeki `data/raw/` klasöründen hem de üst repository
klasöründen otomatik olarak bulunur. Eski veya eksik model artefaktı tespit
edilirse uygulamadaki **Veriyi ve Modeli Hazırla** butonu güvenli biçimde
yeniden eğitim başlatır.

### Testler

```bash
pip install -r requirements-dev.txt
pytest tests -q
```

## 🗂️ Proje Yapısı

```
predictive-retention-ai/
├── data/
│   ├── raw/          # Ham veri (Telco Churn XLS)
│   └── processed/    # Eğitilmiş veri split'leri
├── src/
│   ├── data/         # Veri yükleme ve ön işleme
│   ├── models/       # XGBoost, model karşılaştırma ve kalibrasyon
│   ├── features/     # Segmentasyon ve retention decisioning
│   ├── monitoring/   # Drift ve grup hata analizi
│   ├── operations/   # A/B test, onay, audit ve CRM adaptörü
│   ├── xai/          # SHAP açıklanabilirlik
│   └── llm/          # Groq API entegrasyonu
├── app/
│   └── streamlit_app.py  # 4 sayfalı web arayüzü
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

# Baseline modelleri leakage-safe cross-validation ile karşılaştır
python train.py --compare

# Optimizasyon ve karşılaştırmayı birlikte çalıştır
python train.py --optimize --compare
```

### Streamlit Arayüzü

Uygulama 4 sayfadan oluşur:

1. **📊 Ana Dashboard** — Risk tablosu, segmentler, yönetici özeti, model performansı ve maliyet bazlı eşik
2. **👤 Müşteri Analizi** — CustomerID arama, yeni müşteri tahmini ve SHAP açıklamaları
3. **✉️ Mesaj Üretici** — Groq AI ile düzenlenebilir, indirilebilir ve kopyalanabilir geri kazanım e-postası
4. **Retention Operasyonları** — Kalibrasyon, drift/fairness, A/B uplift, onay/audit ve CRM aktarımı

Next Best Action ekranındaki uplift değerleri başlangıçta açıkça işaretlenmiş
senaryo varsayımlarıdır. Uygulama treatment/control sonuçlarını kaydettikçe
ölçülen retention uplift değerini ayrıca gösterir; churn veri setinden nedensel
etki varmış gibi bir sonuç üretmez.

Model eğitimi aşağıdaki yeniden üretilebilir raporları oluşturur:

- `models/model_metadata.json`
- `models/evaluation_report.json`
- `models/model_comparison.json` (`--compare` kullanıldığında)
- `models/probability_calibrator.pkl`
- `models/calibration_report.json`
- `models/drift_reference.json`

Mesaj ve teklif kayıtları yerel `data/operations.db` SQLite veritabanında
tutulur. CRM webhook aktarımı yalnızca insan tarafından onaylanan kayıtlar için,
arayüzdeki açık gönderme işlemiyle çalışır. Bearer token gerekiyorsa
`.env` içinde `CRM_WEBHOOK_TOKEN` tanımlanabilir.

## 🔑 Groq API Kurulumu

1. [console.groq.com](https://console.groq.com) adresine git
2. Ücretsiz hesap aç → API Key oluştur
3. `.env` dosyasına ekle: `GROQ_API_KEY=gsk_...`

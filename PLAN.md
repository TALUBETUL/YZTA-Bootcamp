# 🧠 Predictive Retention AI — Detaylı Uygulama Planı

## 📌 Proje Özeti

Telekomünikasyon sektöründe müşteri kaybını (churn) önlemek için:
1. **Tahmin** → XGBoost/ANN ile yüksek doğruluklu churn olasılığı
2. **Açıklama** → SHAP/LIME ile "neden gidiyor?" sorusunun cevabı
3. **Kişiselleştirme** → LLM ile empati odaklı geri kazanım metni üretimi
4. **Arayüz** → Streamlit ile karar vericilere yönelik aksiyon panosu

---

## 🗂️ Proje Dizin Yapısı

```
predictive-retention-ai/
│
├── data/
│   ├── raw/                        # Ham veri (WA_Fn-UseC_-Telco-Customer-Churn.xls)
│   ├── processed/                  # İşlenmiş, temizlenmiş veri
│   └── features/                   # Feature engineering çıktıları
│
├── notebooks/
│   ├── 01_eda.ipynb                # Keşifsel Veri Analizi
│   ├── 02_preprocessing.ipynb      # Veri ön işleme
│   ├── 03_model_training.ipynb     # Model eğitimi & karşılaştırma
│   └── 04_xai_analysis.ipynb       # SHAP/LIME analizleri
│
├── src/
│   ├── data/
│   │   ├── loader.py               # Veri yükleme
│   │   └── preprocessor.py         # Temizleme & encoding
│   ├── features/
│   │   └── feature_engineer.py     # Yeni özellik türetme
│   ├── models/
│   │   ├── xgboost_model.py        # XGBoost pipeline
│   │   ├── ann_model.py            # ANN (PyTorch/Keras) pipeline
│   │   └── model_selector.py       # En iyi modeli seç & kaydet
│   ├── xai/
│   │   ├── shap_explainer.py       # SHAP entegrasyonu
│   │   └── lime_explainer.py       # LIME entegrasyonu
│   └── llm/
│       ├── prompt_builder.py       # SHAP sonuçlarından prompt oluştur
│       └── retention_writer.py     # OpenAI / açık kaynak LLM çağrısı
│
├── app/
│   ├── streamlit_app.py            # Ana Streamlit arayüzü
│   ├── components/
│   │   ├── sidebar.py              # Filtreler & ayarlar
│   │   ├── customer_card.py        # Tek müşteri detay kartı
│   │   ├── risk_dashboard.py       # Toplu risk tablosu
│   │   └── retention_message.py    # LLM mesaj gösterimi
│   └── utils/
│       └── chart_helpers.py        # Plotly grafik yardımcıları
│
├── models/                         # Eğitilmiş model dosyaları (.pkl / .pt)
├── requirements.txt
├── .env.example                    # API anahtarı şablonu
└── README.md
```

---

## 📅 Geliştirme Aşamaları

### ✅ Aşama 1 — Veri Hazırlama ve Keşifsel Analiz (EDA)
**Süre:** 1–2 gün

#### 1.1 Veri Yükleme
- `WA_Fn-UseC_-Telco-Customer-Churn.xls` dosyasını yükle
- Sütun tiplerini belirle (kategorik, sayısal, binary)
- Eksik değerleri tespit et

#### 1.2 Keşifsel Veri Analizi
- Churn dağılımı (dengesiz sınıf kontrolü)
- Sayısal değişkenler için histogramlar (tenure, MonthlyCharges, TotalCharges)
- Kategorik değişkenler için churn oranı karşılaştırması (Contract, PaymentMethod, InternetService)
- Korelasyon matrisi (heatmap)
- Churn oranı: ~%26 → sınıf dengesizliği var, SMOTE veya class_weight kullanılacak

#### 1.3 Veri Ön İşleme
- `TotalCharges` sütununu numerige çevir (boş string → NaN → ortalama/medyan ile doldur)
- `customerID` sütununu kaldır
- Binary encoding: `gender`, `Churn`, `Partner`, `Dependents`, vb.
- One-hot encoding: `InternetService`, `Contract`, `PaymentMethod`
- StandardScaler / MinMaxScaler ile normalizasyon

---

### ✅ Aşama 2 — Model Geliştirme ve Değerlendirme
**Süre:** 2–3 gün

#### 2.1 Baseline Modeller

| Model | Kütüphane | Açıklama |
|-------|-----------|----------|
| Logistic Regression | scikit-learn | Baseline |
| Random Forest | scikit-learn | Ağaç tabanlı |
| **XGBoost** | xgboost | Ana model (gradient boosting) |
| **ANN** | PyTorch veya Keras | Derin öğrenme alternatifi |

#### 2.2 Değerlendirme Metrikleri
- **F1-Score** (dengesiz sınıf için kritik)
- **ROC-AUC**
- **Precision-Recall Curve**
- **Confusion Matrix**
- Churn tahmini için **Recall** öncelikli (false negative maliyetli)

#### 2.3 Hiperparametre Optimizasyonu
- XGBoost: `GridSearchCV` veya `Optuna`
- ANN: Learning rate, dropout, katman sayısı

#### 2.4 Model Seçimi & Kaydetme
- En iyi modeli `models/best_model.pkl` olarak kaydet
- `joblib` veya `pickle` kullan

---

### ✅ Aşama 3 — Açıklanabilir Yapay Zeka (XAI)
**Süre:** 1–2 gün

#### 3.1 SHAP Entegrasyonu

```python
import shap

explainer = shap.TreeExplainer(best_model)  # XGBoost için
shap_values = explainer.shap_values(X_test)

# Tek müşteri için waterfall plot
shap.waterfall_plot(shap.Explanation(values=shap_values[i], ...))

# Genel feature importance
shap.summary_plot(shap_values, X_test)
```

#### 3.2 SHAP Çıktılarından Bağlam Üretimi
Her müşteri için en etkili 3–5 faktörü çıkar:
```
Örnek: "Müşteri #1234 için churn riski %87:
  - Aylık ücret artışı: +%23 (yüksek etki)
  - Sözleşme tipi: Aylık (risk artırıcı)
  - Teknik destek başvurusu: 4 kez (yüksek şikayet)"
```

#### 3.3 LIME Alternatifi
- Model tip bağımsız çalışır
- ANN modeli için LIME tercih edilebilir
- `lime.lime_tabular.LimeTabularExplainer`

---

### ✅ Aşama 4 — LLM ile Kişiselleştirilmiş Mesaj Üretimi
**Süre:** 1–2 gün

#### 4.1 Prompt Mühendisliği
SHAP çıktılarından dinamik prompt oluştur:

```python
def build_prompt(customer_data: dict, shap_factors: list) -> str:
    return f"""
    Sen bir telekomünikasyon şirketinin müşteri deneyimi uzmanısın.

    Müşteri Profili:
    - Tenure: {customer_data['tenure']} ay
    - Aylık Ücret: {customer_data['MonthlyCharges']} TL
    - Sözleşme: {customer_data['Contract']}

    Churn Risk Faktörleri (SHAP analizi):
    {chr(10).join([f"- {f}" for f in shap_factors])}

    Bu müşteri için:
    1. Empati dolu, kişiselleştirilmiş bir geri kazanım e-postası yaz.
    2. Somut bir teklif/çözüm öner (indirim, plan değişikliği, öncelikli destek).
    3. Mesaj 3 paragrafı geçmesin, samimi ve doğal olsun.
    """
```

#### 4.2 LLM Seçenekleri

| Seçenek | Avantaj | Dezavantaj |
|---------|---------|------------|
| **OpenAI GPT-4o** | En yüksek kalite | Ücretli API |
| **Gemini 1.5 Flash** | Hızlı, uygun fiyat | — |
| **Ollama (Llama 3)** | Tamamen yerel, ücretsiz | GPU gerektirebilir |
| **Groq API** | Çok hızlı, ücretsiz tier | Rate limit |

#### 4.3 Fallback Mekanizması
- API başarısız olursa → Kural tabanlı şablon mesaj kullan
- `.env` dosyasında API anahtarı yönetimi

---

### ✅ Aşama 5 — Streamlit Arayüzü
**Süre:** 2–3 gün

#### 5.1 Sayfa Yapısı

```
📊 Ana Dashboard
├── 🔴 Toplu Risk Görünümü      → Tüm müşterilerin risk skoru tablosu
├── 👤 Müşteri Detay            → Tek müşteri analizi + SHAP grafiği
└── ✉️ Mesaj Üretici            → LLM ile geri kazanım metni oluştur
```

#### 5.2 Özellikler

**Toplu Risk Tablosu:**
- Filtreleme: Risk skoru > %70, sözleşme tipi, tenure
- Renk kodlaması: 🔴 Yüksek / 🟡 Orta / 🟢 Düşük risk
- CSV export butonu

**Tek Müşteri Analizi:**
- CustomerID gir → anlık tahmin
- SHAP waterfall grafiği (Plotly)
- Risk faktörleri özet kartı

**Geri Kazanım Mesajı:**
- "Mesaj Oluştur" butonu → LLM'e SHAP bağlamı ile istek at
- Üretilen metni düzenle ve kopyala
- Farklı ton seçenekleri: Resmi / Samimi / Kısa & Öz

---

## 🛠️ Teknoloji Yığını (Tech Stack)

| Kategori | Araç | Versiyon |
|----------|------|---------|
| Dil | Python | 3.10+ |
| Veri | Pandas, NumPy | latest |
| ML | Scikit-learn, XGBoost | latest |
| DL | PyTorch veya TensorFlow/Keras | latest |
| XAI | SHAP, LIME | latest |
| LLM | OpenAI / Groq / Ollama | — |
| Arayüz | Streamlit | 1.35+ |
| Görselleştirme | Plotly, Matplotlib, Seaborn | latest |
| Env | python-dotenv | latest |

---

## 📦 requirements.txt İçeriği

```
pandas
numpy
scikit-learn
xgboost
torch
shap
lime
streamlit
plotly
matplotlib
seaborn
openai
python-dotenv
joblib
openpyxl
xlrd
imbalanced-learn
optuna
```

---

## 🔄 Veri Akışı (End-to-End Pipeline)

```
Ham XLS Verisi
      ↓
 Preprocessor
(temizleme, encoding, scaling)
      ↓
Feature Engineering
      ↓
   XGBoost / ANN
   (Tahmin → Churn Olasılığı)
      ↓
  SHAP Explainer
  (Risk Faktörleri)
      ↓
 Prompt Builder
(SHAP → LLM Prompt)
      ↓
   LLM API
(Kişiselleştirilmiş Mesaj)
      ↓
 Streamlit Arayüzü
(Karar Verici Paneli)
```

---

## ⚠️ Riskler ve Önlemler

| Risk | Önlem |
|------|-------|
| Dengesiz sınıf (~%26 churn) | SMOTE veya `class_weight='balanced'` |
| LLM API maliyeti | Groq ücretsiz tier veya Ollama lokal |
| SHAP hesaplama süresi | TreeExplainer (hızlı) veya örnekleme |
| Streamlit performansı | `@st.cache_data` dekoratörleri |
| Veri gizliliği | Gerçek müşteri verisi yerine anonim/sentetik veri |

---

## 🏆 Başarı Kriterleri

- [ ] XGBoost modeli F1 ≥ 0.80 ve ROC-AUC ≥ 0.85
- [ ] SHAP açıklamaları her müşteri için ≤ 2 saniyede üretilmeli
- [ ] LLM mesajı ≤ 10 saniyede gelecek
- [ ] Streamlit arayüzü en az 3 temel sayfayı içermeli
- [ ] README dosyası kurulum ve kullanım talimatlarını içermeli

---

## 📝 Ek Notlar

- Kullanılacak veri seti: `WA_Fn-UseC_-Telco-Customer-Churn.xls` (IBM Telco — 7.043 müşteri, 21 özellik)
- Önce XGBoost ile hızlı bir baseline kur, ardından ANN ile karşılaştır
- LLM entegrasyonu için `.env` dosyasına `OPENAI_API_KEY` veya `GROQ_API_KEY` ekle
- Sunum için SHAP beeswarm plot ve churn dağılım grafiklerini hazır tut

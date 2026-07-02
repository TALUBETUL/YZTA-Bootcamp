Proje Tanımı: Predictive Retention AI
Proje Özeti
Predictive Retention AI, telekomünikasyon sektöründeki müşteri kaybını (churn) önlemek amacıyla geliştirilmiş, uçtan uca bir yapay zeka tabanlı sadakat ve elde tutma sistemidir. Proje, müşteri verilerini analiz ederek potansiyel kayıp risklerini önceden tahmin etmekle kalmaz, aynı zamanda bu risklerin nedensel (causal) açıklamalarını sunar ve LLM (Büyük Dil Modelleri) entegrasyonu ile kişiselleştirilmiş geri kazanım stratejileri üretir.

Projenin Temel Problemi
Geleneksel churn tahmin modelleri genellikle "kara kutu" (black box) mantığıyla çalışır; yani sadece müşterinin gideceğini söyler ancak "neden" gideceğini açıklamaz. Ayrıca, şirketlerin çoğu "tahmin" ile "aksiyon" arasındaki köprüyü kuramaz; riskli müşteriye standart, duygudan yoksun ve etkisiz mesajlar gönderir.

Çözüm Yaklaşımı (Oyun Değiştirici)
Bu proje, churn yönetimini operasyonel bir veri analizinden çıkarıp stratejik bir müşteri deneyimi yönetimine dönüştürür:

Tahmin: Yüksek doğrulukta churn olasılığı hesaplama (XGBoost/ANN).

Açıklanabilirlik (XAI): SHAP/LIME entegrasyonu ile her bir müşteri için "churn risk skoru"nun arkasındaki ana etkenleri (örn: son 3 aydaki fatura artışı, müşteri hizmetleri şikayeti) somutlaştırır.

Kişiselleştirme (LLM): Modelin tespit ettiği risk faktörlerini bir "bağlam" olarak kullanıp, müşteriye özel, empati odaklı ve çözüm sunan geri kazanım metinleri üretir.

Aksiyon Odaklı Arayüz: Tüm analiz ve önerileri tek bir Streamlit arayüzünde birleştirerek, karar vericilerin hızlı aksiyon almasını sağlar.

Kullanılan Teknolojiler
Veri Bilimi: Python (Pandas, Scikit-learn, PyTorch/Keras).

Model Açıklanabilirliği (XAI): SHAP veya LIME.

Üretken Yapay Zeka (LLM): OpenAI API veya Açık Kaynaklı Modeller (Müşteri etkileşimi için).

Web Framework: Streamlit veya Flask.
# YZTA-Bootcamp
YZTA Bootcamp project

## Team Name
(Group 78)

## Proje Geliştiricileri

 - Betül TALU - Scrum Master
 - Serhat AYDIN - Project Manager

## Proje İsmi
Predictive Retention AI

## Backlog Link

https://trello.com/invite/b/6a381148ac23fe2efa8e8fe6/ATTI3d4eebbe8e71dbcd2a8a197810219a7000F4543E/yzta-backlog-board

## Proje Tanımı

Predictive Retention AI, telekomünikasyon sektöründeki müşteri kaybını (churn) önlemek amacıyla geliştirilmiş, uçtan uca bir yapay zeka tabanlı sadakat ve elde tutma sistemidir. Proje, müşteri verilerini analiz ederek potansiyel kayıp risklerini önceden tahmin etmekle kalmaz, aynı zamanda bu risklerin nedensel (causal) açıklamalarını sunar ve LLM (Büyük Dil Modelleri) entegrasyonu ile kişiselleştirilmiş geri kazanım stratejileri üretir.

## Projenin Temel Problemi

Geleneksel churn tahmin modelleri genellikle "kara kutu" (black box) mantığıyla çalışır; yani sadece müşterinin gideceğini söyler ancak "neden" gideceğini açıklamaz. Ayrıca, şirketlerin çoğu "tahmin" ile "aksiyon" arasındaki köprüyü kuramaz; riskli müşteriye standart, duygudan yoksun ve etkisiz mesajlar gönderir.

## Çözüm Yaklaşımı 

Bu proje, churn yönetimini operasyonel bir veri analizinden çıkarıp stratejik bir müşteri deneyimi yönetimine dönüştürür:<br>

Tahmin: Yüksek doğrulukta churn olasılığı hesaplama (XGBoost/ANN).<br>

Açıklanabilirlik (XAI): SHAP/LIME entegrasyonu ile her bir müşteri için "churn risk skoru"nun arkasındaki ana etkenleri (örn: son 3 aydaki fatura artışı, müşteri hizmetleri şikayeti) somutlaştırır.<br>

Kişiselleştirme (LLM): Modelin tespit ettiği risk faktörlerini bir "bağlam" olarak kullanıp, müşteriye özel, empati odaklı ve çözüm sunan geri kazanım metinleri üretir.<br>

Aksiyon Odaklı Arayüz: Tüm analiz ve önerileri tek bir Streamlit arayüzünde birleştirerek, karar vericilerin hızlı aksiyon almasını sağlar.<br>

## Hedef Kitle

Telekomünikasyon operatörleri <br>
SaaS sağlayıcıları <br>
Müşteri İlişkileri (CRM) yöneticileri.<br>


# SPRINTS

<details>
<summary><h3>Sprint1</h3></summary>

## Sprint Goal ve Story Seçimleri

Makine öğrenmesi modelini tamamlamak, performansını değerlendirmek ve açıklanabilir yapay zekâ (XAI) entegrasyonunu gerçekleştirerek kullanıcıya anlamlı tahmin sonuçları sunabilmek.

-Train/Test veri ayrımı	 <br>
-Model seçiminin tamamlanması	 <br>
-XGBoost modelinin eğitilmesi	 <br>
-Model performans metriklerinin hesaplanması (Accuracy, F1, ROC-AUC)	 <br> <br>
-Modelin kaydedilmesi (Joblib)	 <br>
-SHAP TreeExplainer entegrasyonu	 <br>
-SHAP Feature Importance grafikleri	 <br>
-Müşteri bazlı risk faktörlerinin açıklanması	 <br>
-SHAP çıktılarının LLM için uygun formata dönüştürülmesi	 <br>
-Groq API entegrasyonu	 <br>
-Prompt Engineering	 <br>
-Streamlit Dashboard	 <br>
-UI/UX Tasarımı <br>


## Daily Scrum

## 1. Gün

Veri seti projeye aktarıldı. <br>
Yapısal analiz gerçekleştirildi. <br>

## 2. Gün

Missing Value analizi yapıldı. <br>
TotalCharges dönüşümü tamamlandı. <br>

## 3. Gün

Veri ön işleme pipeline'ı oluşturuldu. <br>
Encoding işlemleri tamamlandı. <br>

## 4. Gün

EDA grafikleri oluşturuldu. <br>
Model seçimi üzerine değerlendirmeler yapıldı. <br>

## 5. Gün

Train/Test ayrımı gerçekleştirilmeye başlandı. <br>
İlk model eğitim denemeleri yapıldı. <br>

## Sprint Board Screenshot

<img width="1469" height="800" alt="Screenshot 2026-07-05 at 11 26 06" src="https://github.com/user-attachments/assets/ced79541-508c-415c-8ab5-e53e64a464a4" />

## Sprint Review

Sprint hedeflerine büyük ölçüde ulaşılmıştır.

Model başarıyla eğitilmiş ve değerlendirilmiştir. SHAP entegrasyonu sayesinde model tahminlerinin açıklanabilirliği sağlanmıştır. Böylece kullanıcıların tahmin sonuçlarının neden oluştuğunu görebileceği altyapı hazırlanmıştır.

Bir sonraki sprintte LLM destekli kişiselleştirilmiş mesaj üretimi ve kullanıcı arayüzünün geliştirilmesi planlanmıştır.

## Sprint Retrospective

- Veri hazırlama süreci planlanan takvim doğrultusunda tamamlandı. <br>
- Takım içi görev dağılımı verimli ilerledi. <br>
- Veri ön işleme pipeline'ı sorunsuz şekilde oluşturuldu.<br>
- EDA çıktıları model geliştirme süreci için önemli içgörüler sağladı. <br>
- Performans metriklerinin değerlendirilmesi Sprint 2'ye sarktı. <br>
- Takımın bir araya gelmesinde çıkan sorunlar nedeniyle proje sürecinde sarkmalar yaşandı. <br>

 
</details>


<details>
<summary><h3>Sprint2</h3></summary>
</details>


<details>
<summary><h3>Sprint3</h3></summary>
</details>






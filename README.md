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

## Ürün Özellikleri

Predictive Retention AI, müşteri kaybını önlemeye yönelik yapay zekâ destekli bir karar destek sistemi olarak aşağıdaki temel özellikleri sunmaktadır:

 -Churn Tahmini: Müşterilerin hizmeti bırakma olasılığını makine öğrenmesi modelleri ile tahmin eder.<br>
 -Açıklanabilir Yapay Zekâ (XAI): SHAP analizi sayesinde tahminlerin hangi faktörlere dayandığını kullanıcıya açıklar.<br>
 -LLM Destekli Kişiselleştirilmiş Mesaj Üretimi: Riskli müşteriler için yapay zekâ destekli, kişiselleştirilmiş müşteri tutundurma mesajları oluşturur.<br>
 -İnteraktif Dashboard: Risk dağılımı, müşteri analizleri ve model sonuçlarını görsel olarak sunar.<br>
 -Müşteri Bazlı Analiz: Seçilen bir müşterinin churn riski ve bu riske etki eden faktörleri detaylı olarak gösterir.<br>
 -Yönetici Özet Raporu: Riskli müşteri grupları için özet analizler ve karar destek bilgileri sağlar.<br>
 -CSV Dışa Aktarma: Analiz sonuçlarının CSV formatında dışa aktarılmasını destekler.<br>
 -Modern Kullanıcı Arayüzü: Streamlit tabanlı, sade ve kullanıcı dostu bir arayüz ile kolay kullanım sunar.<br>

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
-Model performans metriklerinin hesaplanması (Accuracy, F1, ROC-AUC)	 <br>
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

<img width="1470" height="802" alt="Screenshot 2026-07-05 at 12 57 32" src="https://github.com/user-attachments/assets/ac51c2e0-4a9c-49e1-9b7e-1a075d48f61c" />

## Ürün Durumu

Sprint 1 sonunda projenin temel veri hazırlama süreci başarıyla tamamlanmıştır. Telco Customer Churn veri seti sisteme aktarılmış, veri temizleme ve ön işleme adımları uygulanmış, eksik veriler giderilmiş ve gerekli veri dönüşümleri gerçekleştirilmiştir. Ayrıca gerçekleştirilen Keşifsel Veri Analizi (EDA) ile veri setinin yapısı, müşteri davranışları ve churn ile ilişkili temel değişkenler analiz edilmiştir.

Bu sprint sonunda makine öğrenmesi modeli için gerekli veri altyapısı hazırlanmış olup model geliştirme çalışmalarına başlanmıştır.

### Gerçekleştirilen Özellikler

-  Veri setinin sisteme yüklenmesi <br>
-  Veri temizleme işlemleri <br>
-  Veri ön işleme (Preprocessing Pipeline) <br>
-  Keşifsel Veri Analizi (EDA) <br>
-  Veri görselleştirmeleri <br>
-  Train/Test veri ayrımı <br>
-  İlk model geliştirme çalışmaları <br>

<img width="942" height="637" alt="Screenshot 2026-07-06 at 14 42 19" src="https://github.com/user-attachments/assets/e5a39e3d-adc7-4f57-9e37-82ad094f2a74" />

<img width="1262" height="629" alt="Screenshot 2026-07-06 at 14 44 24" src="https://github.com/user-attachments/assets/c9ad886b-4559-4171-ab71-e7908d551c5a" />


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

## Sprint Goal ve Story Seçimleri

Makine öğrenmesi modelinin performansını artırmak, modeli açıklanabilir hale getirmek ve uygulamanın temel kullanıcı arayüzünü oluşturmaya başlamak.
Sprint 1 sonunda veri hazırlama ve ön işleme süreci tamamlandığından, Sprint 2 kapsamında ürünün yapay zekâ katmanının geliştirilmesine odaklanılmıştır. Bu sprintte öncelik; churn tahmin modelinin optimize edilmesi, model çıktılarının açıklanabilir hale getirilmesi ve uygulamanın temel ekranlarının geliştirilmesine verilmiştir.

- XGBoost modelinin geliştirilmesi ve optimize edilmesi <br>
- Model performansının değerlendirilmesi <br>
- SHAP entegrasyonunun gerçekleştirilmesi <br>
- SHAP Feature Importance ve görselleştirmelerinin hazırlanması <br>
- SHAP çıktılarının LLM'e uygun hale getirilmesi <br>
- Streamlit uygulamasının oluşturulması <br>
- Müşteri analiz ekranının geliştirilmesi <br>
- Mesaj üretim ekranının geliştirilmesi <br>
- Dashboard geliştirmelerine başlanması <br>
- Prompt Engineering çalışmalarına başlanması <br>
- UI/UX tasarım çalışmalarına başlanması <br>

## Daily Scrum

## 1. Gün

Sprint planlaması gerçekleştirildi. <br>
XGBoost modeli optimize edilmeye başlandı. <br>

## 2. Gün

Model performans metrikleri değerlendirildi. <br>
SHAP entegrasyon çalışmaları başlatıldı. <br>

## 3. Gün

SHAP Feature Importance ve görselleştirmeleri oluşturuldu. <br>
Risk faktörlerinin yorumlanması üzerinde çalışıldı. <br>

## 4. Gün

Streamlit uygulamasının temel sayfaları geliştirildi. <br>
Müşteri analiz ekranı oluşturuldu. <br>

## 5. Gün

Dashboard tasarımı ve kullanıcı arayüzü geliştirmeleri sürdürüldü. <br>
Prompt Engineering çalışmaları devam etti. <br>

## Sprint Board Screenshot

<img width="1470" height="800" alt="Screenshot 2026-07-18 at 21 43 55" src="https://github.com/user-attachments/assets/e4323c94-399d-4367-9eef-8574bae3baf0" />

## Ürün Durumu

Sprint 2 sonunda ürünün makine öğrenmesi modeli geliştirilmiş ve performans değerlendirmeleri tamamlanmıştır. Ayrıca model tahminlerini açıklamak amacıyla SHAP entegrasyonu gerçekleştirilmiş ve temel görselleştirmeler hazırlanmıştır.

Bunun yanında Streamlit tabanlı uygulamanın temel altyapısı oluşturulmuş, müşteri analiz ekranı ile mesaj üretim ekranı geliştirilmiştir. Dashboard tasarımı, kullanıcı arayüzü iyileştirmeleri ve prompt geliştirme çalışmaları ise bir sonraki sprintte tamamlanacaktır.

### Gerçekleştirilen Özellikler

-  Churn tahmin modeli <br>
-  Model performans değerlendirmeleri <br>
-  SHAP tabanlı açıklanabilir yapay zekâ <br>
-  SHAP görselleştirmeleri <br>
-  Streamlit uygulama altyapısı <br>
-  Müşteri analiz ekranı <br>
-  Mesaj üretim ekranı <br>

******* GÖRSEL EKLENECEK ******

## Sprint Review

Sprint hedeflerinin büyük bölümü başarıyla tamamlanmıştır. Model geliştirme ve açıklanabilir yapay zekâ entegrasyonu planlandığı şekilde ilerlemiş, ürünün temel Streamlit altyapısı oluşturulmuştur. Dashboard tasarımı ve kullanıcı deneyimi geliştirmeleri devam etmekte olup, bu çalışmaların Sprint 3 içerisinde tamamlanması planlanmaktadır.

## Sprint Retrospective

  -  Güçlü Yönler <br>
- Model geliştirme süreci planlandığı şekilde ilerledi. <br>
- SHAP entegrasyonu başarıyla tamamlandı. <br>
- Streamlit uygulamasının temel yapısı oluşturuldu. <br>
- Takım içi iletişim düzenli şekilde sürdürüldü. <br>
  -  Karşılaşılan Zorluklar <br>
- Dashboard tasarımı beklenenden daha fazla geliştirme süresi gerektirdi. <br>
- Prompt Engineering sürecinde farklı senaryolar için ek denemeler yapıldı. <br>

</details>


<details>
<summary><h3>Sprint3</h3></summary>
</details>






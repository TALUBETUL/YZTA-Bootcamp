"""
LLM Prompt Oluşturma Modülü
SHAP faktörlerini ve müşteri verilerini birleştirerek LLM prompt'u oluşturur.
"""


RETENTION_SYSTEM_PROMPT = """Sen deneyimli bir telekomünikasyon şirketi müşteri deneyimi uzmanısın.
Görevin, yapay zeka destekli analizler sonucunda tespit edilen yüksek churn riskli müşteriler için
kişiselleştirilmiş, empati odaklı ve somut çözümler sunan geri kazanım e-postaları yazmak.

Yazım kuralları:
- Türkçe yaz, samimi ve sıcak bir ton kullan
- Müşteriyi ismiyle selamla (verilmişse)
- Sorunu kabul et, özür dile ve çözüm sun
- 2-3 paragrafı geçme
- Somut bir teklif ekle (indirim, plan yükseltme, öncelikli destek vb.)
- Jargon kullanma, anlaşılır ol
- Müşteriyi manipüle etme, dürüst ve şeffaf ol"""


def build_retention_prompt(customer_info: dict,
                            shap_factors: list[dict],
                            churn_probability: float,
                            tone: str = "samimi") -> str:
    """
    Müşteri bilgileri ve SHAP faktörlerinden LLM prompt'u oluşturur.

    Args:
        customer_info: Müşteri bilgileri (tenure, MonthlyCharges, Contract, vb.)
        shap_factors: get_top_factors() çıktısı
        churn_probability: Churn olasılığı (0-1 arası)
        tone: "samimi" | "resmi" | "kısa"

    Returns:
        str: LLM'e gönderilecek kullanıcı prompt'u
    """
    # Müşteri profili özeti
    tenure_months = customer_info.get("tenure", "bilinmiyor")
    monthly_charge = customer_info.get("MonthlyCharges", "bilinmiyor")
    contract = customer_info.get("Contract", "bilinmiyor")
    customer_name = customer_info.get("name", "Sayın Müşterimiz")

    # SHAP faktörlerini metne çevir
    factors_text = "\n".join([
        f"  - {f['feature']}: {f['direction']} (etki: {abs(f['shap_value']):.3f})"
        for f in shap_factors
    ])

    # Ton ayarları
    tone_instructions = {
        "samimi": "Samimi, sıcak ve arkadaşça bir ton kullan.",
        "resmi": "Profesyonel ve kurumsal bir ton kullan.",
        "kısa": "Çok kısa tut, maksimum 2 cümle başlık + 1 paragraf teklif.",
    }
    tone_note = tone_instructions.get(tone, tone_instructions["samimi"])

    prompt = f"""Aşağıdaki müşteri için kişiselleştirilmiş bir geri kazanım e-postası yaz.

MÜŞTERİ BİLGİLERİ:
- İsim: {customer_name}
- Abonelik Süresi: {tenure_months} ay
- Aylık Ücret: {monthly_charge} TL
- Sözleşme Tipi: {contract}
- Churn Risk Skoru: %{churn_probability*100:.1f}

CHURN RİSK FAKTÖRLERİ (Yapay Zeka Analizi):
{factors_text}

TON: {tone_note}

Lütfen doğrudan e-posta metnini yaz (konu satırı dahil). Başka açıklama ekleme."""

    return prompt


def build_batch_summary_prompt(high_risk_customers: list[dict]) -> str:
    """
    Toplu yüksek riskli müşteri listesi için yönetici özeti prompt'u oluşturur.

    Returns:
        str: Yönetici özeti için LLM prompt'u
    """
    customer_lines = []
    for i, c in enumerate(high_risk_customers[:10], 1):  # Max 10 müşteri
        customer_lines.append(
            f"  {i}. Müşteri ID: {c.get('id', '?')} | "
            f"Risk: %{c.get('churn_prob', 0)*100:.1f} | "
            f"Tenure: {c.get('tenure', '?')} ay | "
            f"Aylık: {c.get('MonthlyCharges', '?')} TL"
        )

    prompt = f"""Aşağıdaki {len(high_risk_customers)} yüksek riskli müşteri listesi için
yöneticiye sunulacak kısa bir Türkçe özet rapor hazırla.
Ortak risk örüntülerini belirt ve 2-3 aksiyon önerisi sun.

YÜKSEK RİSKLİ MÜŞTERİLER:
{chr(10).join(customer_lines)}

Özet raporu 3-4 paragrafta yaz."""

    return prompt

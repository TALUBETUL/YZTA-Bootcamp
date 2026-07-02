"""
LLM Geri Kazanım Mesajı Üretici
Groq API (llama-3.3-70b) kullanarak kişiselleştirilmiş mesajlar üretir.
"""

import os
from groq import Groq
from dotenv import load_dotenv
from src.llm.prompt_builder import RETENTION_SYSTEM_PROMPT


load_dotenv()


def get_groq_client() -> Groq:
    """Groq API client'ını başlatır."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY bulunamadı. Lütfen .env dosyasına ekleyin.\n"
            "Ücretsiz API key için: https://console.groq.com"
        )
    return Groq(api_key=api_key)


def generate_retention_message(user_prompt: str,
                                 system_prompt: str = RETENTION_SYSTEM_PROMPT,
                                 model: str = "llama-3.3-70b-versatile",
                                 max_tokens: int = 500,
                                 temperature: float = 0.7) -> str:
    """
    Groq API üzerinden LLM ile geri kazanım mesajı üretir.

    Args:
        user_prompt: build_retention_prompt() çıktısı
        system_prompt: Sistem rolü talimatı
        model: Kullanılacak Groq modeli
        max_tokens: Maksimum token sayısı
        temperature: Yaratıcılık seviyesi (0-1)

    Returns:
        str: Üretilen e-posta metni
    """
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        message = response.choices[0].message.content.strip()
        return message

    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg or "authentication" in error_msg.lower():
            return _fallback_message("API anahtarı hatası. Lütfen GROQ_API_KEY'i kontrol edin.")
        elif "rate limit" in error_msg.lower():
            return _fallback_message("Rate limit aşıldı. Lütfen birkaç saniye bekleyip tekrar deneyin.")
        else:
            return _fallback_message(f"Bir hata oluştu: {error_msg}")


def _fallback_message(error_detail: str) -> str:
    """
    LLM başarısız olduğunda şablon mesaj döndürür.
    """
    return f"""[⚠️ LLM Bağlantı Hatası — Şablon Mesaj]

{error_detail}

---

Konu: Sizi Kaybetmek İstemiyoruz — Özel Teklifimiz

Sayın Değerli Müşterimiz,

Uzun süredir güvenilir bir müşterimiz olduğunuz için sizinle daha iyi bir deneyim oluşturmak istiyoruz.
Son dönemde yaşadığınız sorunların farkındayız ve bu nedenle size özel bir teklif hazırladık.

Bir sonraki 3 aylık fatura döneminde %20 indirim ve öncelikli teknik destek hizmeti sunmaktan memnuniyet duyarız.
Teklifimizden yararlanmak için müşteri hizmetlerimizi arayabilir veya bu e-postayı yanıtlayabilirsiniz.

Saygılarımızla,
Müşteri Deneyimi Ekibi"""


def generate_batch_summary(user_prompt: str,
                            model: str = "llama-3.3-70b-versatile") -> str:
    """
    Toplu yönetici özet raporu üretir.
    """
    return generate_retention_message(
        user_prompt=user_prompt,
        system_prompt="Sen bir iş analisti ve müşteri deneyimi danışmanısın. Yöneticilere kısa, net ve uygulanabilir raporlar hazırlarsın.",
        model=model,
        max_tokens=800,
        temperature=0.5,
    )

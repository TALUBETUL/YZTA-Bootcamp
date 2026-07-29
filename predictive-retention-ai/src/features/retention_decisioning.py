"""Next-best-action and transparent retention economics."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RetentionAction:
    code: str
    name: str
    reason: str
    offer_cost: float
    expected_uplift: float


def estimate_customer_value(customer: dict, horizon_months: int = 12) -> float:
    """Simple gross revenue-at-risk proxy; intentionally not a profit forecast."""
    monthly = max(float(customer.get("MonthlyCharges", 0) or 0), 0)
    tenure = max(float(customer.get("tenure", 0) or 0), 0)
    loyalty_multiplier = min(1.25, 1 + tenure / 720)
    return round(monthly * max(horizon_months, 1) * loyalty_multiplier, 2)


def candidate_actions(customer: dict) -> list[RetentionAction]:
    actions = [
        RetentionAction(
            "priority_support",
            "Öncelikli teknik destek",
            "Hızlı sorun çözümü ve proaktif destek sunar.",
            12.0,
            0.04,
        )
    ]
    if customer.get("Contract") == "Month-to-month":
        actions.append(RetentionAction(
            "annual_contract",
            "Yıllık sözleşmeye geçiş teşviki",
            "Aylık sözleşme kaynaklı riski azaltmayı hedefler.",
            35.0,
            0.10,
        ))
    if customer.get("PaymentMethod") == "Electronic check":
        actions.append(RetentionAction(
            "autopay",
            "Otomatik ödeme teşviki",
            "Ödeme deneyimini sadeleştirir.",
            15.0,
            0.05,
        ))
    if customer.get("InternetService") == "Fiber optic":
        actions.append(RetentionAction(
            "fiber_service_review",
            "Fiber hizmet kalite incelemesi",
            "Fiber müşterilerinde hizmet deneyimini kontrol eder.",
            25.0,
            0.07,
        ))
    if customer.get("TechSupport") in {"No", 0, False}:
        actions.append(RetentionAction(
            "tech_support_trial",
            "3 aylık teknik destek denemesi",
            "Destek erişimi eksikliğini giderir.",
            24.0,
            0.08,
        ))
    return actions


def score_action(
    action: RetentionAction,
    customer: dict,
    churn_probability: float,
    horizon_months: int = 12,
) -> dict:
    customer_value = estimate_customer_value(customer, horizon_months)
    probability = min(max(float(churn_probability), 0), 1)
    scenario_uplift = min(action.expected_uplift, probability)
    expected_benefit = scenario_uplift * customer_value
    net_value = expected_benefit - action.offer_cost
    return {
        **asdict(action),
        "customer_value": customer_value,
        "churn_probability": probability,
        "scenario_uplift": scenario_uplift,
        "expected_benefit": round(expected_benefit, 2),
        "expected_net_value": round(net_value, 2),
        "profitable": net_value > 0,
        "evidence_level": "scenario_assumption",
    }


def recommend_next_best_action(
    customer: dict,
    churn_probability: float,
    horizon_months: int = 12,
) -> dict:
    """Rank eligible actions by transparent expected net value."""
    scored = [
        score_action(action, customer, churn_probability, horizon_months)
        for action in candidate_actions(customer)
    ]
    scored.sort(key=lambda item: item["expected_net_value"], reverse=True)
    best = scored[0]
    if best["expected_net_value"] <= 0:
        return {
            "code": "no_paid_offer",
            "name": "Ücretli teklif gönderme",
            "reason": "Mevcut varsayımlarla pozitif beklenen net değer oluşmadı.",
            "offer_cost": 0.0,
            "customer_value": best["customer_value"],
            "churn_probability": best["churn_probability"],
            "scenario_uplift": 0.0,
            "expected_benefit": 0.0,
            "expected_net_value": 0.0,
            "profitable": True,
            "evidence_level": "scenario_assumption",
            "alternatives": scored,
        }
    return {**best, "alternatives": scored}

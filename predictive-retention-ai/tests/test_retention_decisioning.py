from src.features.retention_decisioning import (
    estimate_customer_value, recommend_next_best_action,
)


def test_next_best_action_exposes_economic_assumptions():
    customer = {
        "MonthlyCharges": 100,
        "tenure": 24,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "InternetService": "Fiber optic",
        "TechSupport": "No",
    }
    action = recommend_next_best_action(customer, 0.8)

    assert action["customer_value"] == estimate_customer_value(customer)
    assert action["evidence_level"] == "scenario_assumption"
    assert action["alternatives"]
    assert action["expected_net_value"] == max(
        item["expected_net_value"] for item in action["alternatives"]
    )

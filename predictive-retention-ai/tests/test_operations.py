import pytest

from src.operations.crm import approved_records_dataframe, send_to_crm_webhook
from src.operations.governance import OperationsStore, validate_retention_message


def test_safety_blocks_sensitive_requests_and_excessive_discount():
    result = validate_retention_message(
        "Kredi kartı numarası verin, size %50 indirim garanti."
    )
    assert not result["safe_to_approve"]
    assert {"sensitive_data_request", "discount_limit"} <= {
        issue["code"] for issue in result["issues"]
    }

    offer_result = validate_retention_message(
        "Size %10 indirim sunuyoruz.",
        approved_offer={"name": "Öncelikli teknik destek"},
    )
    assert not offer_result["safe_to_approve"]


def test_approval_audit_experiment_and_crm_gate(tmp_path):
    store = OperationsStore(tmp_path / "operations.db")
    safety = validate_retention_message("Konu: Size özel destek teklifimiz")
    record_id = store.create_message(
        "C-1", "Konu: Size özel destek teklifimiz",
        {"code": "support", "name": "Destek", "offer_cost": 10},
        safety,
    )
    store.review_message(record_id, "approved", "reviewer@example.com")
    approved = store.list_messages("approved")

    assert approved[0]["id"] == record_id
    assert len(store.list_audit_events()) == 2
    assert len(approved_records_dataframe(approved)) == 1
    with pytest.raises(ValueError):
        send_to_crm_webhook(
            {**approved[0], "status": "draft"}, "https://crm.example/hook"
        )

    first = store.assign_variant("campaign", "C-1")
    second = store.assign_variant("campaign", "C-1")
    assert first == second
    store.record_outcome("campaign", "C-1", retained=True)
    assert store.experiment_summary("campaign")[first]["retained"] == 1


def test_unsafe_message_cannot_be_approved(tmp_path):
    store = OperationsStore(tmp_path / "operations.db")
    safety = validate_retention_message("CVV ve şifre verin.")
    record_id = store.create_message("C-2", "CVV ve şifre verin.", {}, safety)
    with pytest.raises(ValueError):
        store.review_message(record_id, "approved", "reviewer")

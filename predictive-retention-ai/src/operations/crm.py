"""Explicit, approval-gated CRM export helpers."""

from __future__ import annotations

import json
import os
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd


def approved_records_dataframe(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        if record.get("status") != "approved":
            continue
        offer = record.get("offer", {})
        rows.append({
            "record_id": record["id"],
            "customer_id": record["customer_id"],
            "message": record["message"],
            "offer_code": offer.get("code"),
            "offer_name": offer.get("name"),
            "offer_cost": offer.get("offer_cost"),
            "expected_net_value": offer.get("expected_net_value"),
            "approved_by": record.get("reviewer"),
            "approved_at": record.get("updated_at"),
        })
    return pd.DataFrame(rows)


def send_to_crm_webhook(record: dict, webhook_url: str, timeout: int = 10) -> int:
    """Send one approved record to an HTTPS webhook after an explicit UI action."""
    if record.get("status") != "approved":
        raise ValueError("Yalnızca onaylanmış kayıtlar CRM'e gönderilebilir.")
    parsed = urlparse(webhook_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("CRM webhook adresi geçerli bir HTTPS URL olmalıdır.")
    headers = {"Content-Type": "application/json"}
    token = os.getenv("CRM_WEBHOOK_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = json.dumps(record, ensure_ascii=False, default=str).encode("utf-8")
    request = Request(webhook_url, data=payload, headers=headers, method="POST")
    with urlopen(request, timeout=timeout) as response:
        return int(response.status)

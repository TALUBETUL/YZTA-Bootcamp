"""LLM safety checks plus auditable message/offer approval storage."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "operations.db"


def validate_retention_message(
    message: str,
    allowed_discount: float = 30,
    approved_offer: dict | None = None,
) -> dict:
    issues = []
    text = (message or "").strip()
    if not text:
        issues.append({"severity": "error", "code": "empty", "message": "Mesaj boş."})
    if len(text) > 2500:
        issues.append({"severity": "warning", "code": "length", "message": "Mesaj çok uzun."})
    if re.search(r"\b(?:şifre|parola|kredi kartı numarası|cvv)\b", text, re.I):
        issues.append({
            "severity": "error",
            "code": "sensitive_data_request",
            "message": "Mesaj hassas bilgi talep ediyor.",
        })
    if re.search(r"\b(?:kesinlikle|garanti|%100)\b", text, re.I):
        issues.append({
            "severity": "warning",
            "code": "unsupported_guarantee",
            "message": "Doğrulanmamış garanti ifadesi olabilir.",
        })
    if approved_offer is not None:
        allowed_discount = float(approved_offer.get("discount_percent", 0) or 0)
    for match in re.finditer(r"%\s*(\d+(?:[.,]\d+)?)", text):
        discount = float(match.group(1).replace(",", "."))
        if discount > allowed_discount:
            issues.append({
                "severity": "error",
                "code": "discount_limit",
                "message": f"%{discount:g} indirim, %{allowed_discount:g} sınırını aşıyor.",
            })
    if approved_offer and approved_offer.get("name"):
        offer_words = {
            word.casefold()
            for word in re.findall(r"\w+", str(approved_offer["name"]))
            if len(word) >= 5
        }
        normalized_message = text.casefold()
        if offer_words and not any(word in normalized_message for word in offer_words):
            issues.append({
                "severity": "warning",
                "code": "offer_not_identifiable",
                "message": "Mesajdaki teklif, onaylı aksiyonla açıkça eşleştirilemiyor.",
            })
    if "LLM Bağlantı Hatası" in text:
        issues.append({
            "severity": "error",
            "code": "fallback_message",
            "message": "API hata metni müşteriye gönderilemez.",
        })
    return {
        "safe_to_approve": not any(item["severity"] == "error" for item in issues),
        "issues": issues,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


class OperationsStore:
    def __init__(self, path: str | Path = DEFAULT_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS campaign_assignments (
                    campaign_id TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    assigned_at TEXT NOT NULL,
                    outcome TEXT,
                    outcome_at TEXT,
                    PRIMARY KEY (campaign_id, customer_id)
                );
                CREATE TABLE IF NOT EXISTS message_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    offer_json TEXT NOT NULL,
                    safety_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    reviewer TEXT
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    def assign_variant(self, campaign_id: str, customer_id: str) -> str:
        digest = hashlib.sha256(f"{campaign_id}:{customer_id}".encode()).digest()
        variant = "treatment" if digest[0] % 2 else "control"
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO campaign_assignments
                   (campaign_id, customer_id, variant, assigned_at)
                   VALUES (?, ?, ?, ?)""",
                (campaign_id, customer_id, variant, now),
            )
            row = connection.execute(
                """SELECT variant FROM campaign_assignments
                   WHERE campaign_id=? AND customer_id=?""",
                (campaign_id, customer_id),
            ).fetchone()
        return row["variant"]

    def record_outcome(self, campaign_id: str, customer_id: str, retained: bool):
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE campaign_assignments SET outcome=?, outcome_at=?
                   WHERE campaign_id=? AND customer_id=?""",
                ("retained" if retained else "churned", self._now(), campaign_id, customer_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Önce müşteri için deney varyantı atanmalıdır.")

    def experiment_summary(self, campaign_id: str) -> dict:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT variant, outcome, COUNT(*) AS n
                   FROM campaign_assignments WHERE campaign_id=?
                   GROUP BY variant, outcome""",
                (campaign_id,),
            ).fetchall()
        summary = {
            "campaign_id": campaign_id,
            "control": {"assigned": 0, "measured": 0, "retained": 0},
            "treatment": {"assigned": 0, "measured": 0, "retained": 0},
        }
        for row in rows:
            bucket = summary[row["variant"]]
            bucket["assigned"] += row["n"]
            if row["outcome"]:
                bucket["measured"] += row["n"]
                if row["outcome"] == "retained":
                    bucket["retained"] += row["n"]
        for bucket in (summary["control"], summary["treatment"]):
            bucket["retention_rate"] = (
                bucket["retained"] / bucket["measured"] if bucket["measured"] else None
            )
        control_rate = summary["control"]["retention_rate"]
        treatment_rate = summary["treatment"]["retention_rate"]
        summary["measured_uplift"] = (
            treatment_rate - control_rate
            if treatment_rate is not None and control_rate is not None else None
        )
        return summary

    def create_message(self, customer_id: str, message: str, offer: dict, safety: dict) -> int:
        now = self._now()
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO message_records
                   (customer_id, message, offer_json, safety_json, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'draft', ?, ?)""",
                (
                    customer_id,
                    message,
                    json.dumps(offer, ensure_ascii=False),
                    json.dumps(safety, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            record_id = int(cursor.lastrowid)
            self._audit(connection, "message", str(record_id), "created", "system", {})
        return record_id

    def review_message(self, record_id: int, status: str, reviewer: str):
        if status not in {"approved", "rejected"}:
            raise ValueError("Durum approved veya rejected olmalıdır.")
        reviewer = reviewer.strip()
        if not reviewer:
            raise ValueError("Onaylayan/reddeden kişi belirtilmelidir.")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT safety_json, status FROM message_records WHERE id=?", (record_id,)
            ).fetchone()
            if row is None:
                raise ValueError("Kayıt bulunamadı.")
            safety = json.loads(row["safety_json"])
            if status == "approved" and not safety.get("safe_to_approve"):
                raise ValueError("Güvenlik hataları olan mesaj onaylanamaz.")
            connection.execute(
                """UPDATE message_records SET status=?, reviewer=?, updated_at=?
                   WHERE id=?""",
                (status, reviewer, self._now(), record_id),
            )
            self._audit(
                connection, "message", str(record_id), status, reviewer,
                {"previous_status": row["status"]},
            )

    def list_messages(self, status: str | None = None) -> list[dict]:
        query = "SELECT * FROM message_records"
        params = ()
        if status:
            query += " WHERE status=?"
            params = (status,)
        query += " ORDER BY id DESC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._decode_message(row) for row in rows]

    def list_audit_events(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events ORDER BY id DESC"
            ).fetchall()
        return [
            {**dict(row), "details": json.loads(row["details_json"])}
            for row in rows
        ]

    def _audit(self, connection, entity_type, entity_id, action, actor, details):
        connection.execute(
            """INSERT INTO audit_events
               (entity_type, entity_id, action, actor, details_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                entity_type, entity_id, action, actor,
                json.dumps(details, ensure_ascii=False), self._now(),
            ),
        )

    @staticmethod
    def _decode_message(row):
        item = dict(row)
        item["offer"] = json.loads(item.pop("offer_json"))
        item["safety"] = json.loads(item.pop("safety_json"))
        return item

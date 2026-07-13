"""
Map OpenAI Privacy Filter span labels to Presidio entity types used by VARIOS AI.

Detection is done exclusively by the privacy-filter model (8 native labels).
This module only renames model spans to Presidio entity_type values — no regex
re-detection or rule-based recognizers.

Presidio entities in VARIOS AI:
CREDIT_CARD, IBAN_CODE, CRYPTO, PHONE_NUMBER, EMAIL_ADDRESS, PERSON,
IP_ADDRESS, NRP, LOCATION
"""
from __future__ import annotations

from typing import Any

# Hier is die Liste der erlaubten Entities
PRESIDIO_ENTITIES: frozenset[str] = frozenset(
    {
        "CREDIT_CARD",
        "CRYPTO",
        #DATETIME
        "EMAIL_ADDRESS",
        "IBAN_CODE",
        "IP_ADDRESS",
        #MAC ADDRESS
        "NRP",
        "LOCATION",
        "PERSON",
        "PHONE_NUMBER",
        #MEDICAL LICENSE
        #URL
    }
)

# All eight native OpenAI privacy-filter labels → Presidio (None = no matching Presidio entity).
_OPF_LABEL_MAP: dict[str, str | None] = {
    "private_person": "PERSON",
    "private_address": "LOCATION",
    "private_email": "EMAIL_ADDRESS",
    "private_phone": "PHONE_NUMBER",
    "account_number": "CREDIT_CARD",
    "secret": "CRYPTO",
    # Model detects URLs/dates, but Presidio has no URL/DATE entity in VARIOS AI.
    "private_url": None,
    "private_date": None,
}

# Hier ist die mapping Funktion
def map_opf_to_presidio(label: str, span_text: str) -> str | None:
    """Rename a model span label to a Presidio entity_type (label map only)."""
    key = (label or "").lower().strip()
    mapped = _OPF_LABEL_MAP.get(key)
    if mapped is None and key not in _OPF_LABEL_MAP:
        return None
    return mapped


def entity_allowed(entity_type: str | None, requested_entities: list[str]) -> bool:
    if entity_type is None or entity_type not in PRESIDIO_ENTITIES:
        return False
    if requested_entities and entity_type not in requested_entities:
        return False
    return True


def make_result(
    start: int,
    end: int,
    entity_type: str,
    score: float,
    recognizer: str,
    *,
    original_label: str | None = None,
) -> dict[str, Any]:
    explanation: dict[str, Any] = {
        "recognizer": recognizer,
        "pattern": None,
    }
    if original_label is not None:
        explanation["original_label"] = original_label
    return {
        "start": start,
        "end": end,
        "entity_type": entity_type,
        "score": score,
        "analysis_explanation": explanation,
    }

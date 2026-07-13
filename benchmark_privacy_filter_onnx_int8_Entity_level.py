"""Entity-level benchmark for openai/privacy-filter (PyTorch or ONNX INT8).

Reports strict, typed entity recognition metrics.  A true positive requires the
same Presidio entity type *and* exactly identical character boundaries.  This
is the standard strict NER criterion; overlapping spans are not counted as a
match.  Model and dataset labels are mapped to the entity vocabulary defined
in entity_mapping.py.  Labels outside that vocabulary are reported separately.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from entity_mapping import PRESIDIO_ENTITIES, map_opf_to_presidio
from benchmark_privacy_filter_onnx_int8 import (
    align_mbert_tokens_to_text,
    load_tokenizer_robust,
    load_validation_dataset,
    resolve_device,
)


@dataclass(frozen=True, order=True)
class Entity:
    start: int
    end: int
    entity_type: str


@dataclass
class EntityMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else 0.0

    @property
    def support(self) -> int:
        return self.tp + self.fn


# Dataset aliases are deliberately explicit.  They cover the actual OpenPII
# classes and preserve the coarser target taxonomy of entity_mapping.py.  Types
# such as DATE, TAXNUM, PASSPORTNUM, and SOCIALNUM have no semantically valid
# target in that taxonomy and therefore deliberately remain unscored.
_REFERENCE_ALIASES = {
    "PERSON": "PERSON", "NAME": "PERSON", "FIRST_NAME": "PERSON", "LAST_NAME": "PERSON",
    "GIVENNAME": "PERSON", "SURNAME": "PERSON", "TITLE": "PERSON", "PRIVATE_PERSON": "PERSON",
    "ADDRESS": "LOCATION", "LOCATION": "LOCATION", "STREET": "LOCATION", "BUILDINGNUM": "LOCATION",
    "CITY": "LOCATION", "STATE": "LOCATION", "COUNTRY": "LOCATION", "ZIP_CODE": "LOCATION",
    "ZIPCODE": "LOCATION", "POSTCODE": "LOCATION", "PRIVATE_ADDRESS": "LOCATION", "EMAIL": "EMAIL_ADDRESS",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS", "PRIVATE_EMAIL": "EMAIL_ADDRESS", "PHONE": "PHONE_NUMBER",
    "PHONE_NUMBER": "PHONE_NUMBER", "TELEPHONE": "PHONE_NUMBER", "TELEPHONENUM": "PHONE_NUMBER", "PRIVATE_PHONE": "PHONE_NUMBER",
    "CREDIT_CARD": "CREDIT_CARD", "CREDIT_CARD_NUMBER": "CREDIT_CARD", "CREDITCARDNUMBER": "CREDIT_CARD",
    "ACCOUNT_NUMBER": "CREDIT_CARD", "ACCOUNTNUMBER": "CREDIT_CARD", "IBAN": "IBAN_CODE", "IBAN_CODE": "IBAN_CODE",
    "IP": "IP_ADDRESS", "IP_ADDRESS": "IP_ADDRESS", "CRYPTO": "CRYPTO",
    "CRYPTO_WALLET": "CRYPTO", "SECRET": "CRYPTO", "NRP": "NRP",
    "NATIONALITY": "NRP", "RELIGION": "NRP", "POLITICAL_AFFILIATION": "NRP",
}

# These are the only Presidio types reachable from a native privacy-filter
# label.  The remaining values in PRESIDIO_ENTITIES are retained in the table
# for transparency, but cannot produce a TP without changing entity_mapping.py.
_MODEL_SUPPORTED_ENTITIES = frozenset(
    {"CREDIT_CARD", "CRYPTO", "EMAIL_ADDRESS", "LOCATION", "PERSON", "PHONE_NUMBER"}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict typed entity-level benchmark for openai/privacy-filter.")
    parser.add_argument("--model", default="openai/privacy-filter")
    parser.add_argument("--dataset", default="ai4privacy/open-pii-masking-500k-ai4privacy")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--filter-language", default="de")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--runtime", choices=["torch", "onnx-int8"], default="torch")
    parser.add_argument("--output-path", default="benchmark_privacy_filter_onnx_int8_entity_level_results.txt")
    return parser.parse_args()


def strip_bio(label: object) -> tuple[str | None, str]:
    """Return (BIO prefix, normalized raw label), accepting BIO/BILOU labels."""
    raw = str(label or "").strip().upper().replace("-", "_")
    if raw in {"", "O", "0", "NONE", "NON_PII", "NONPII"}:
        return None, ""
    for prefix in ("B_", "I_", "L_", "U_", "S_", "E_"):
        if raw.startswith(prefix):
            return prefix[0], raw[len(prefix):]
    return None, raw


def map_label(label: object, *, model_label: bool) -> str | None:
    _, raw = strip_bio(label)
    if not raw:
        return None
    # The production mapping is authoritative for native privacy-filter labels.
    if model_label:
        mapped = map_opf_to_presidio(raw.lower(), "")
        if mapped is not None:
            return mapped
    return _REFERENCE_ALIASES.get(raw)


def entities_from_labeled_offsets(
    offsets: Iterable[tuple[int, int] | None], labels: Iterable[object], *, model_label: bool,
) -> tuple[list[Entity], Counter[str]]:
    """Convert BIO/BILOU token labels into typed character spans.

    Adjacent I/L tokens of the same type continue an entity even when their
    offsets are separated by whitespace.  A malformed I token starts a new
    entity, which avoids discarding model output while keeping it auditable.
    """
    result: list[Entity] = []
    unsupported: Counter[str] = Counter()
    current: Entity | None = None

    def finish() -> None:
        nonlocal current
        if current is not None:
            result.append(current)
            current = None

    for offset, label in zip(offsets, labels):
        prefix, raw = strip_bio(label)
        entity_type = map_label(label, model_label=model_label)
        if offset is None or entity_type is None:
            if raw and entity_type is None:
                unsupported[raw] += 1
            finish()
            continue
        start, end = offset
        if end <= start:
            continue
        continuation = (
            current is not None
            and current.entity_type == entity_type
            and (
                # BIO/BILOU continuation may legitimately span a separating space.
                prefix in {"I", "L", "E"}
                # Some token classifiers expose plain type labels.  In that case,
                # only join contiguous subword offsets; do not join separate words.
                or (prefix is None and start <= current.end)
            )
        )
        if continuation:
            current = Entity(current.start, end, entity_type)
        else:
            finish()
            current = Entity(start, end, entity_type)
        if prefix in {"L", "E", "U", "S"}:
            finish()
    finish()
    return result, unsupported


def predict_entities_torch(texts: list[str], tokenizer: AutoTokenizer, model: AutoModelForTokenClassification, device: torch.device):
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")
    inputs = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        prediction_ids = model(**inputs).logits.argmax(dim=-1).cpu()
    return entities_from_predictions(encoded["input_ids"], offsets, prediction_ids, tokenizer, model.config.id2label)


def predict_entities_onnx(texts: list[str], tokenizer: AutoTokenizer, session: ort.InferenceSession, id2label: dict[int, str]):
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")
    inputs = {"input_ids": encoded["input_ids"].numpy().astype(np.int64, copy=False), "attention_mask": encoded["attention_mask"].numpy().astype(np.int64, copy=False)}
    if "token_type_ids" in encoded:
        inputs["token_type_ids"] = encoded["token_type_ids"].numpy().astype(np.int64, copy=False)
    prediction_ids = np.argmax(session.run(None, inputs)[0], axis=-1)
    return entities_from_predictions(encoded["input_ids"], offsets, prediction_ids, tokenizer, id2label)


def entities_from_predictions(input_ids, offsets, prediction_ids, tokenizer, id2label):
    batch: list[tuple[list[Entity], Counter[str]]] = []
    for ids, sample_offsets, classes in zip(input_ids, offsets, prediction_ids):
        usable_offsets = [None if int(token.item()) in tokenizer.all_special_ids else (int(start), int(end)) for token, (start, end) in zip(ids, sample_offsets)]
        labels = [id2label[int(class_id)] for class_id in classes]
        batch.append(entities_from_labeled_offsets(usable_offsets, labels, model_label=True))
    return batch


def update_metrics(metrics: EntityMetrics, truth: set[Entity], predicted: set[Entity]) -> None:
    metrics.tp += len(truth & predicted)
    metrics.fp += len(predicted - truth)
    metrics.fn += len(truth - predicted)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    args = parse_args()
    dataset = load_validation_dataset(args.dataset, args.split)
    if args.filter_language and "language" in dataset.column_names:
        language = args.filter_language.strip().lower()
        dataset = dataset.filter(lambda row: str(row.get("language", "")).lower() == language)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    tokenizer = load_tokenizer_robust(args.model)
    device = resolve_device(args.device)
    if args.runtime == "onnx-int8":
        if device.type != "cpu":
            raise ValueError("ONNX INT8 wird nur auf CPU unterstützt; nutze --device cpu.")
        from benchmark_privacy_filter_onnx_int8 import build_onnx_int8_session
        session, model = build_onnx_int8_session(args.model), None
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        id2label = {int(key): value for key, value in config.id2label.items()}
    else:
        model = AutoModelForTokenClassification.from_pretrained(args.model, trust_remote_code=True).to(device)
        model.eval()
        session, id2label = None, None

    by_type = defaultdict(EntityMetrics)
    overall = EntityMetrics()
    unsupported_truth: Counter[str] = Counter()
    unsupported_prediction: Counter[str] = Counter()
    non_empty = evaluable = 0
    batch_rows: list[dict] = []

    def flush() -> None:
        nonlocal evaluable
        if not batch_rows:
            return
        texts = [row["source_text"] for row in batch_rows]
        predicted_batch = (predict_entities_onnx(texts, tokenizer, session, id2label) if session else predict_entities_torch(texts, tokenizer, model, device))
        for row, (predicted, unsupported_pred) in zip(batch_rows, predicted_batch):
            offsets = align_mbert_tokens_to_text(row["source_text"], row["mbert_tokens"])
            truth, unsupported_gt = entities_from_labeled_offsets(offsets, row["mbert_token_classes"], model_label=False)
            unsupported_truth.update(unsupported_gt)
            unsupported_prediction.update(unsupported_pred)
            # Unsupported reference labels cannot be scored against the selected taxonomy.
            if truth or predicted:
                evaluable += 1
            truth_set, predicted_set = set(truth), set(predicted)
            update_metrics(overall, truth_set, predicted_set)
            for entity_type in PRESIDIO_ENTITIES:
                update_metrics(by_type[entity_type], {x for x in truth_set if x.entity_type == entity_type}, {x for x in predicted_set if x.entity_type == entity_type})
        batch_rows.clear()

    for row in tqdm(dataset, total=len(dataset), desc="Entity benchmark", unit="sample", dynamic_ncols=True):
        text, tokens, labels = row.get("source_text") or "", row.get("mbert_tokens"), row.get("mbert_token_classes")
        if not text:
            continue
        non_empty += 1
        if not (isinstance(tokens, list) and isinstance(labels, list) and len(tokens) == len(labels)):
            raise ValueError("Entity-level evaluation requires aligned mbert_tokens and mbert_token_classes.")
        batch_rows.append(row)
        if len(batch_rows) >= max(1, args.batch_size):
            flush()
    flush()

    supported_types = [entity_type for entity_type in PRESIDIO_ENTITIES if by_type[entity_type].support]
    macro_f1 = sum(by_type[entity_type].f1 for entity_type in supported_types) / len(supported_types) if supported_types else 0.0
    lines = [
        "==== Privacy Filter: Strict Typed Entity-Level Benchmark ====",
        f"Model:                       {args.model}", f"Runtime:                     {args.runtime}",
        f"Dataset / split:             {args.dataset} / {args.split}", f"Language filter:             {args.filter_language or '(none)'}",
        f"Rows / non-empty rows:       {len(dataset):,} / {non_empty:,}", f"Rows with scored entities:   {evaluable:,}",
        "Matching:                    exact character boundaries + identical mapped entity type",
        "Model-supported target types: " + ", ".join(sorted(_MODEL_SUPPORTED_ENTITIES)),
        "Unsupported model target types: " + ", ".join(sorted(PRESIDIO_ENTITIES - _MODEL_SUPPORTED_ENTITIES)),
        f"Micro precision / recall / F1: {pct(overall.precision)} / {pct(overall.recall)} / {pct(overall.f1)}",
        f"Macro F1 ({len(supported_types)} types with reference support): {pct(macro_f1)}", "", "Per entity type (strict):",
        "entity_type       support      TP      FP      FN   precision    recall        F1",
    ]
    for entity_type in sorted(PRESIDIO_ENTITIES):
        m = by_type[entity_type]
        lines.append(f"{entity_type:<18} {m.support:>7,} {m.tp:>7,} {m.fp:>7,} {m.fn:>7,} {pct(m.precision):>10} {pct(m.recall):>9} {pct(m.f1):>9}")
    lines.append("\nUnsupported reference labels (not scored): " + (", ".join(f"{k}={v}" for k, v in sorted(unsupported_truth.items())) or "none"))
    lines.append("Unsupported model labels (not scored):     " + (", ".join(f"{k}={v}" for k, v in sorted(unsupported_prediction.items())) or "none"))
    Path(args.output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Results file: {args.output_path}")


if __name__ == "__main__":
    main()

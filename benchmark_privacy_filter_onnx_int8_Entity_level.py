"""Entity-level benchmark for openai/privacy-filter (PyTorch or ONNX INT8).

Reports native-model-class entity recognition metrics. A true positive
requires the same privacy-filter class and overlapping character boundaries.
Dataset labels are mapped to the model's eight native classes, so GIVENNAME,
SURNAME, and TITLE, for example, are all scored as PRIVATE_PERSON. Labels with
no matching model class are reported separately.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from benchmark_privacy_filter_onnx_int8 import (
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
    # At entity-span level there is no finite universe of negative spans.  TN
    # therefore counts rows in which this entity type is absent in both truth
    # and prediction (a sample-level TN).
    tn: int = 0

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


# Map detailed OpenPII reference labels to the privacy-filter's native output
# classes. This is deliberately a coarse mapping: every name component is a
# PRIVATE_PERSON, and every address component is a PRIVATE_ADDRESS.
_REFERENCE_TO_MODEL_CLASS = {
    "DATE": "PRIVATE_DATE",
    "PRIVATE_DATE": "PRIVATE_DATE",
    "PERSON": "PRIVATE_PERSON",
    "NAME": "PRIVATE_PERSON",
    "FIRST_NAME": "PRIVATE_PERSON",
    "LAST_NAME": "PRIVATE_PERSON",
    "GIVENNAME": "PRIVATE_PERSON",
    "SURNAME": "PRIVATE_PERSON",
    "TITLE": "PRIVATE_PERSON",
    "PRIVATE_PERSON": "PRIVATE_PERSON",
    "EMAIL": "PRIVATE_EMAIL",
    "EMAIL_ADDRESS": "PRIVATE_EMAIL",
    "PRIVATE_EMAIL": "PRIVATE_EMAIL",
    "PHONE": "PRIVATE_PHONE",
    "PHONE_NUMBER": "PRIVATE_PHONE",
    "TELEPHONE": "PRIVATE_PHONE",
    "TELEPHONENUM": "PRIVATE_PHONE",
    "PRIVATE_PHONE": "PRIVATE_PHONE",
    "ADDRESS": "PRIVATE_ADDRESS",
    "LOCATION": "PRIVATE_ADDRESS",
    "STREET": "PRIVATE_ADDRESS",
    "BUILDINGNUM": "PRIVATE_ADDRESS",
    "CITY": "PRIVATE_ADDRESS",
    "STATE": "PRIVATE_ADDRESS",
    "COUNTRY": "PRIVATE_ADDRESS",
    "ZIP_CODE": "PRIVATE_ADDRESS",
    "ZIPCODE": "PRIVATE_ADDRESS",
    "POSTCODE": "PRIVATE_ADDRESS",
    "PRIVATE_ADDRESS": "PRIVATE_ADDRESS",
    "CREDIT_CARD": "ACCOUNT_NUMBER",
    "CREDIT_CARD_NUMBER": "ACCOUNT_NUMBER",
    "CREDITCARDNUMBER": "ACCOUNT_NUMBER",
    "ACCOUNT_NUMBER": "ACCOUNT_NUMBER",
    "ACCOUNTNUMBER": "ACCOUNT_NUMBER",
    # The privacy-filter's ACCOUNT_NUMBER class also covers document, social,
    # and tax identification numbers in this benchmark taxonomy.
    "IDCARDNUM": "ACCOUNT_NUMBER",
    "PASSPORTNUM": "ACCOUNT_NUMBER",
    "SOCIALNUM": "ACCOUNT_NUMBER",
    "TAXNUM": "ACCOUNT_NUMBER",
    "PRIVATE_URL": "PRIVATE_URL",
    "URL": "PRIVATE_URL",
    "SECRET": "SECRET",
}

_MODEL_CLASSES = frozenset(
    {
        "ACCOUNT_NUMBER",
        "PRIVATE_ADDRESS",
        "PRIVATE_DATE",
        "PRIVATE_EMAIL",
        "PRIVATE_PERSON",
        "PRIVATE_PHONE",
        "PRIVATE_URL",
        "SECRET",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict typed entity-level benchmark for openai/privacy-filter.")
    parser.add_argument("--model", default="openai/privacy-filter")
    parser.add_argument("--dataset", default="ai4privacy/pii-masking-openpii-1m")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--filter-language", default="de")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--runtime", choices=["torch", "onnx-int8"], default="onnx-int8")
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
    if model_label:
        return raw if raw in _MODEL_CLASSES else None
    return _REFERENCE_TO_MODEL_CLASS.get(raw)


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

    # Token labels are reduced to entity spans here.  All downstream scoring
    # therefore operates on original character positions, not token indices.
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


def entities_from_privacy_mask(raw_entities: object) -> tuple[list[Entity], Counter[str]]:
    """Read the dataset's canonical, typed character spans.

    ``privacy_mask`` already contains the annotated start/end offsets.  Using
    it avoids trying to reconstruct reference character spans from mBERT
    wordpieces, which can be ambiguous for repeated text and unknown tokens.
    """
    if raw_entities is None:
        return [], Counter()
    if isinstance(raw_entities, str):
        raw_entities = ast.literal_eval(raw_entities)
    if not isinstance(raw_entities, list):
        raise ValueError("privacy_mask must be a list of annotated entity dictionaries.")

    result: list[Entity] = []
    unsupported: Counter[str] = Counter()
    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        _, raw_label = strip_bio(item.get("label"))
        entity_type = map_label(raw_label, model_label=False)
        if entity_type is None:
            if raw_label:
                unsupported[raw_label] += 1
            continue
        try:
            start, end = int(item["start"]), int(item["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid privacy_mask entity: {item!r}") from exc
        if end > start:
            result.append(Entity(start, end, entity_type))
    return result, unsupported


def predict_entities_torch(texts: list[str], tokenizer: AutoTokenizer, model: AutoModelForTokenClassification, device: torch.device):
    # Tokenize the complete batch consistently with the model.  Offset mappings
    # retain the link from each subword token back to its source-text span.
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")
    inputs = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        # Logits contain one score per token and class; argmax selects the
        # model's most likely label independently for every token.
        prediction_ids = model(**inputs).logits.argmax(dim=-1).cpu()
    # Convert predicted token classes into comparable typed character entities.
    return entities_from_predictions(encoded["input_ids"], offsets, prediction_ids, tokenizer, model.config.id2label)


def predict_entities_onnx(texts: list[str], tokenizer: AutoTokenizer, session: ort.InferenceSession, id2label: dict[int, str]):
    # Use the same tokenizer and offsets as the PyTorch path so that runtime
    # choice cannot change how predictions are mapped back to the text.
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")
    inputs = {"input_ids": encoded["input_ids"].numpy().astype(np.int64, copy=False), "attention_mask": encoded["attention_mask"].numpy().astype(np.int64, copy=False)}
    if "token_type_ids" in encoded:
        inputs["token_type_ids"] = encoded["token_type_ids"].numpy().astype(np.int64, copy=False)
    # ONNX returns the logits array; selecting its largest class score yields
    # the token-level prediction equivalent to the PyTorch argmax above.
    prediction_ids = np.argmax(session.run(None, inputs)[0], axis=-1)
    return entities_from_predictions(encoded["input_ids"], offsets, prediction_ids, tokenizer, id2label)


def entities_from_predictions(input_ids, offsets, prediction_ids, tokenizer, id2label):
    batch: list[tuple[list[Entity], Counter[str]]] = []
    for ids, sample_offsets, classes in zip(input_ids, offsets, prediction_ids):
        # Special tokens such as [CLS], [SEP], and padding have no source-text
        # counterpart and must not produce entities.  Remaining tokenizer
        # offsets map predicted subwords directly to character boundaries.
        usable_offsets = [None if int(token.item()) in tokenizer.all_special_ids else (int(start), int(end)) for token, (start, end) in zip(ids, sample_offsets)]
        labels = [id2label[int(class_id)] for class_id in classes]
        # BIO/BILOU decoding joins compatible subwords into final entity spans.
        batch.append(entities_from_labeled_offsets(usable_offsets, labels, model_label=True))
    return batch


def update_metrics(metrics: EntityMetrics, truth: set[Entity], predicted: set[Entity]) -> None:
    """Update model-class coverage metrics using typed span overlap.

    The OpenPII ground truth splits names and addresses into detailed pieces,
    while privacy-filter can return one coarser span. A reference entity is a
    TP when any prediction of the same native model class overlaps it. A
    prediction is an FP only when it overlaps no reference of its class.
    """
    def overlaps_same_class(left: Entity, right: Entity) -> bool:
        return (
            left.entity_type == right.entity_type
            and left.start < right.end
            and right.start < left.end
        )

    matched_truth = {item for item in truth if any(overlaps_same_class(item, candidate) for candidate in predicted)}
    matched_prediction = {item for item in predicted if any(overlaps_same_class(item, candidate) for candidate in truth)}
    metrics.tp += len(matched_truth)
    metrics.fp += len(predicted) - len(matched_prediction)
    metrics.fn += len(truth) - len(matched_truth)
    metrics.tn += int(not truth and not predicted)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    # Arguemnte laden
    args = parse_args()
    #Datensatz einladen und filtern
    dataset = load_validation_dataset(args.dataset, args.split)
    if args.filter_language and "language" in dataset.column_names:
        language = args.filter_language.strip().lower()
        dataset = dataset.filter(lambda row: str(row.get("language", "")).lower() == language)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    #Tokenizer einladen
    tokenizer = load_tokenizer_robust(args.model)
    device = resolve_device(args.device)
    #CPU Nutzung
    if args.runtime == "onnx-int8":
        if device.type != "cpu":
            raise ValueError("ONNX INT8 wird nur auf CPU unterstützt; nutze --device cpu.")
        from benchmark_privacy_filter_onnx_int8 import build_onnx_int8_session
        session, model = build_onnx_int8_session(args.model), None
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        id2label = {int(key): value for key, value in config.id2label.items()}
    #GPU Nutzung
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
        # Run inference in batches to reduce model overhead while preserving a
        # separate entity list for every source row.
        predicted_batch = (predict_entities_onnx(texts, tokenizer, session, id2label) if session else predict_entities_torch(texts, tokenizer, model, device))
        for row, (predicted, unsupported_pred) in zip(batch_rows, predicted_batch):
            # The dataset supplies canonical, typed character spans in
            # privacy_mask.  These are the reference positions against which
            # tokenizer-derived model spans must be compared.
            truth, unsupported_gt = entities_from_privacy_mask(row.get("privacy_mask"))
            unsupported_truth.update(unsupported_gt)
            unsupported_prediction.update(unsupported_pred)
            # Unsupported reference labels cannot be scored against the selected taxonomy.
            if truth or predicted:
                evaluable += 1
            truth_set, predicted_set = set(truth), set(predicted)
            # Update the global (micro) contingency totals over all entities.
            update_metrics(overall, truth_set, predicted_set)
            for entity_type in _MODEL_CLASSES:
                # Build the per-class TP/FP/FN matrix by restricting both span
                # sets to one target entity type before applying strict match.
                update_metrics(by_type[entity_type], {x for x in truth_set if x.entity_type == entity_type}, {x for x in predicted_set if x.entity_type == entity_type})
        batch_rows.clear()

    for row in tqdm(dataset, total=len(dataset), desc="Entity benchmark", unit="sample", dynamic_ncols=True):
        text, tokens, labels = row.get("source_text") or "", row.get("mbert_tokens"), row.get("mbert_token_classes")
        if not text:
            continue
        non_empty += 1
        if not (isinstance(tokens, list) and isinstance(labels, list) and len(tokens) == len(labels)):
            raise ValueError("Entity-level evaluation requires aligned mbert_tokens and mbert_token_classes.")
        #Hier wird der Eintrag in eine liste angehangen und damit ein Batch gebaut
        batch_rows.append(row)
        if len(batch_rows) >= max(1, args.batch_size): # prüfen ob der batch fertig gebaut ist, sonst noch einen Eintrag rein
            flush()
    flush()

    # Macro F1 gives every reference-supported model class equal weight;
    # micro metrics above instead aggregate TP/FP/FN over all entity spans.
    supported_types = [entity_type for entity_type in _MODEL_CLASSES if by_type[entity_type].support]
    macro_f1 = sum(by_type[entity_type].f1 for entity_type in supported_types) / len(supported_types) if supported_types else 0.0
    lines = [
        "==== Privacy Filter: Strict Typed Entity-Level Benchmark ====",
        f"Model:                       {args.model}", f"Runtime:                     {args.runtime}",
        f"Dataset / split:             {args.dataset} / {args.split}", f"Language filter:             {args.filter_language or '(none)'}",
        f"Benchmark entries:           {len(dataset):,}",
        "Matching:                    overlapping privacy_mask span + identical model class",
        "Model classes:               " + ", ".join(sorted(_MODEL_CLASSES)),
        f"Micro precision / recall / F1: {pct(overall.precision)} / {pct(overall.recall)} / {pct(overall.f1)}",
        f"Macro F1 ({len(supported_types)} classes with reference support): {pct(macro_f1)}", "", "Per model class (class-aware overlap):",
        "model_class       support      TP      FP      FN  TN (rows)   precision    recall        F1",
    ]
    for entity_type in sorted(_MODEL_CLASSES):
        m = by_type[entity_type]
        lines.append(f"{entity_type:<18} {m.support:>7,} {m.tp:>7,} {m.fp:>7,} {m.fn:>7,} {m.tn:>10,} {pct(m.precision):>10} {pct(m.recall):>9} {pct(m.f1):>9}")
    # These are annotated entity occurrences, not token-label occurrences.
    lines.append("\nUnsupported reference entities (not scored): " + (", ".join(f"{k}={v}" for k, v in sorted(unsupported_truth.items())) or "none"))
    lines.append("Unsupported model label tokens (not scored):     " + (", ".join(f"{k}={v}" for k, v in sorted(unsupported_prediction.items())) or "none"))
    # Persist the complete metric table so a benchmark run remains auditable
    # independently of the console output.
    Path(args.output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Results file: {args.output_path}")


if __name__ == "__main__":
    main()

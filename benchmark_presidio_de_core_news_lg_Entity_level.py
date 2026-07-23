"""Entity-level benchmark for Presidio using spaCy ``de_core_news_lg``.

The scoring uses the *same output taxonomy* as
``benchmark_privacy_filter_onnx_int8_Entity_level.py``.  Both models are thus
evaluated against the same mapped reference labels and the same eight output
classes.  A reference entity is detected when an appropriately mapped Presidio
result overlaps its character span.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from tqdm import tqdm

import benchmark_privacy_filter_onnx_int8_Entity_level as entity_benchmark
from benchmark_privacy_filter_onnx_int8 import load_validation_dataset


# Translate Presidio's native recognizer outputs to the privacy-filter's
# output taxonomy.  This makes a micro/macro score comparable to the ONNX INT8
# report without pretending that both systems have identical recognizers.
_PRESIDIO_TO_COMPARISON_CLASS = {
    "CREDIT_CARD": "ACCOUNT_NUMBER",
    "IBAN_CODE": "ACCOUNT_NUMBER",
    "ID": "ACCOUNT_NUMBER",
    "DATE_TIME": "PRIVATE_DATE",
    "EMAIL": "PRIVATE_EMAIL",
    "EMAIL_ADDRESS": "PRIVATE_EMAIL",
    "LOCATION": "PRIVATE_ADDRESS",
    "PERSON": "PRIVATE_PERSON",
    "PHONE_NUMBER": "PRIVATE_PHONE",
    "URL": "PRIVATE_URL",
}

_COMPARISON_CLASSES = entity_benchmark._MODEL_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict typed entity-level benchmark for Presidio with spaCy de_core_news_lg."
    )
    parser.add_argument("--dataset", default="ai4privacy/pii-masking-openpii-1m")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--filter-language", default="de")
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument(
        "--output-path", default="benchmark_presidio_de_core_news_lg_entity_level_results.txt"
    )
    return parser.parse_args()


def build_analyzer() -> AnalyzerEngine:
    """Create Presidio with exactly the requested German spaCy model."""
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "de", "model_name": "de_core_news_lg"}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    return AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["de"])


def map_reference_label(label: object) -> str | None:
    """Use exactly the reference-label mapping of the ONNX INT8 benchmark."""
    return entity_benchmark.map_label(label, model_label=False)


def entities_from_privacy_mask(raw_entities: object):
    """Read canonical reference offsets, retaining unsupported-label counts."""
    # The shared helper has the desired robust parsing and Entity representation.
    # Temporarily using its mapping globals would make concurrent imports fragile,
    # so retain the small parsing loop locally.
    import ast

    if raw_entities is None:
        return [], Counter()
    if isinstance(raw_entities, str):
        raw_entities = ast.literal_eval(raw_entities)
    if not isinstance(raw_entities, list):
        raise ValueError("privacy_mask must be a list of annotated entity dictionaries.")

    entities, unsupported = [], Counter()
    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        _, raw_label = entity_benchmark.strip_bio(item.get("label"))
        entity_type = map_reference_label(raw_label)
        if entity_type is None:
            if raw_label:
                unsupported[raw_label] += 1
            continue
        try:
            start, end = int(item["start"]), int(item["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid privacy_mask entity: {item!r}") from exc
        if end > start:
            entities.append(entity_benchmark.Entity(start, end, entity_type))
    return entities, unsupported


def predict_entities(analyzer: AnalyzerEngine, text: str, threshold: float):
    """Convert Presidio character-span results to the shared Entity format."""
    entities, unsupported = [], Counter()
    for result in analyzer.analyze(text=text, language="de", score_threshold=threshold):
        presidio_type = str(result.entity_type).upper()
        entity_type = _PRESIDIO_TO_COMPARISON_CLASS.get(presidio_type)
        if entity_type is None:
            unsupported[presidio_type] += 1
            continue
        if result.end > result.start:
            entities.append(entity_benchmark.Entity(result.start, result.end, entity_type))
    return entities, unsupported


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    args = parse_args()
    analyzer = build_analyzer()
    dataset = load_validation_dataset(args.dataset, args.split)
    if args.filter_language and "language" in dataset.column_names:
        language = args.filter_language.strip().lower()
        if language != "de":
            raise ValueError("This benchmark is fixed to de_core_news_lg; use --filter-language de.")
        dataset = dataset.filter(lambda row: str(row.get("language", "")).lower() == language)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    by_type = defaultdict(entity_benchmark.EntityMetrics)
    overall = entity_benchmark.EntityMetrics()
    unsupported_truth: Counter[str] = Counter()
    unsupported_prediction: Counter[str] = Counter()
    non_empty = evaluable = 0

    for row in tqdm(dataset, total=len(dataset), desc="Entity benchmark", unit="sample", dynamic_ncols=True):
        text = row.get("source_text") or ""
        if not text:
            continue
        non_empty += 1
        truth, unsupported_gt = entities_from_privacy_mask(row.get("privacy_mask"))
        predicted, unsupported_pred = predict_entities(analyzer, text, args.score_threshold)
        unsupported_truth.update(unsupported_gt)
        unsupported_prediction.update(unsupported_pred)
        if truth or predicted:
            evaluable += 1

        truth_set, predicted_set = set(truth), set(predicted)
        entity_benchmark.update_metrics(overall, truth_set, predicted_set)
        for entity_type in _COMPARISON_CLASSES:
            entity_benchmark.update_metrics(
                by_type[entity_type],
                {x for x in truth_set if x.entity_type == entity_type},
                {x for x in predicted_set if x.entity_type == entity_type},
            )

    supported_types = [kind for kind in _COMPARISON_CLASSES if by_type[kind].support]
    macro_f1 = (
        sum(by_type[kind].f1 for kind in supported_types) / len(supported_types)
        if supported_types
        else 0.0
    )
    lines = [
        "==== Presidio (de_core_news_lg): Privacy-Filter-Comparable Entity Benchmark ====",
        "Model:                       de_core_news_lg (Presidio spaCy NLP engine)",
        f"Score threshold:             {args.score_threshold}",
        f"Dataset / split:             {args.dataset} / {args.split}",
        f"Language filter:             {args.filter_language or '(none)'}",
        f"Benchmark entries:           {len(dataset):,}",
        f"Non-empty entries:           {non_empty:,}",
        f"Evaluable entries:           {evaluable:,}",
        "Matching:                    overlapping privacy_mask span + identical comparison class",
        "Comparison classes:          " + ", ".join(sorted(_COMPARISON_CLASSES)),
        f"Micro precision / recall / F1: {pct(overall.precision)} / {pct(overall.recall)} / {pct(overall.f1)}",
        f"Macro F1 ({len(supported_types)} classes with reference support): {pct(macro_f1)}",
        "",
        "Per comparison class (class-aware overlap):",
        "comparison_class   support      TP      FP      FN  TN (rows)   precision    recall        F1",
    ]
    for entity_type in sorted(_COMPARISON_CLASSES):
        metrics = by_type[entity_type]
        lines.append(
            f"{entity_type:<18} {metrics.support:>7,} {metrics.tp:>7,} {metrics.fp:>7,} "
            f"{metrics.fn:>7,} {metrics.tn:>10,} {pct(metrics.precision):>10} "
            f"{pct(metrics.recall):>9} {pct(metrics.f1):>9}"
        )
    lines.append(
        "\nUnsupported reference entities (not scored): "
        + (", ".join(f"{key}={value}" for key, value in sorted(unsupported_truth.items())) or "none")
    )
    lines.append(
        "Unsupported Presidio entities (no comparison class): "
        + (", ".join(f"{key}={value}" for key, value in sorted(unsupported_prediction.items())) or "none")
    )
    Path(args.output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Results file: {args.output_path}")


if __name__ == "__main__":
    main()

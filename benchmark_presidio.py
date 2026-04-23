import argparse
import ast
import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider


@dataclass
class TokenMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, y_true: list[bool], y_pred: list[bool]) -> None:
        for truth, pred in zip(y_true, y_pred):
            if pred and truth:
                self.tp += 1
            elif pred and not truth:
                self.fp += 1
            elif truth and not pred:
                self.fn += 1

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0


@dataclass
class SpanMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, truth_spans: set[tuple[int, int]], pred_spans: set[tuple[int, int]]) -> None:
        self.tp += len(truth_spans & pred_spans)
        self.fp += len(pred_spans - truth_spans)
        self.fn += len(truth_spans - pred_spans)

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return (2 * self.tp) / denom if denom else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Presidio (en_core_web_trf) on ai4privacy/pii-masking-300k validation split."
    )
    parser.add_argument("--dataset", default="ai4privacy/pii-masking-300k")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--language", default="de")
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument(
        "--filter-language",
        default="",
        help="Filter auf eine Datensatzsprache, z. B. english oder german.",
    )
    parser.add_argument(
        "--filter-english",
        action="store_true",
        help="Filtert auf language == English (empfohlen für en_core_web_trf).",
    )
    return parser.parse_args()


def build_analyzer() -> AnalyzerEngine:
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "de", "model_name": "de_core_news_lg"}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["de"])


def load_validation_dataset(dataset_name: str, split_name: str):
    try:
        return load_dataset(dataset_name, split=split_name)
    except Exception:
        # Fallback for datasets that keep split info in a "set" column.
        train_ds = load_dataset(dataset_name, split="train")
        if "set" not in train_ds.column_names:
            raise
        filtered = train_ds.filter(lambda row: str(row.get("set", "")).lower() == split_name.lower())
        if len(filtered) == 0:
            raise ValueError(f"Split '{split_name}' not found in dataset '{dataset_name}'.")
        return filtered


def parse_span_labels(raw_span_labels) -> list[tuple[int, int]]:
    if raw_span_labels is None:
        return []
    if isinstance(raw_span_labels, str):
        raw_span_labels = ast.literal_eval(raw_span_labels)

    spans: list[tuple[int, int]] = []
    for item in raw_span_labels:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        start, end = int(item[0]), int(item[1])
        if end > start:
            spans.append((start, end))
    return merge_spans(spans)


def merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(spans, key=lambda x: (x[0], x[1]))
    if not ordered:
        return []

    merged: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def extract_token_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def mark_tokens_as_pii(token_spans: list[tuple[int, int]], pii_spans: list[tuple[int, int]]) -> list[bool]:
    if not pii_spans:
        return [False] * len(token_spans)

    marks = [False] * len(token_spans)
    span_index = 0
    ordered_spans = sorted(pii_spans, key=lambda x: (x[0], x[1]))

    for idx, (tok_start, tok_end) in enumerate(token_spans):
        while span_index < len(ordered_spans) and ordered_spans[span_index][1] <= tok_start:
            span_index += 1
        check_index = span_index
        while check_index < len(ordered_spans) and ordered_spans[check_index][0] < tok_end:
            span_start, span_end = ordered_spans[check_index]
            if span_end > tok_start and span_start < tok_end:
                marks[idx] = True
                break
            check_index += 1
    return marks


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    args = parse_args()
    analyzer = build_analyzer()
    dataset = load_validation_dataset(args.dataset, args.split)

    if args.filter_language and "language" in dataset.column_names:
        lang = args.filter_language.strip().lower()
        dataset = dataset.filter(lambda row: str(row.get("language", "")).lower() == lang)

    if args.filter_english and "language" in dataset.column_names:
        dataset = dataset.filter(lambda row: str(row.get("language", "")).lower() == "english")

    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    token_metrics = TokenMetrics()
    span_metrics = SpanMetrics()

    for row in dataset:
        text = row.get("source_text") or ""
        if not text:
            continue

        gt_spans = parse_span_labels(row.get("span_labels"))
        analysis = analyzer.analyze(
            text=text,
            language=args.language,
            score_threshold=args.score_threshold,
        )
        pred_spans = merge_spans((result.start, result.end) for result in analysis)

        tokens = extract_token_spans(text)
        y_true = mark_tokens_as_pii(tokens, gt_spans)
        y_pred = mark_tokens_as_pii(tokens, pred_spans)
        token_metrics.update(y_true, y_pred)

        span_metrics.update(set(gt_spans), set(pred_spans))

    print("==== Presidio Benchmark (ai4privacy/pii-masking-300k) ====")
    print(f"Samples: {len(dataset)}")
    print(f"Scope Precision (tokens): {format_pct(token_metrics.precision)}")
    print(f"Recall (tokens):          {format_pct(token_metrics.recall)}")
    print(f"F1 (tokens):              {format_pct(token_metrics.f1)}")
    print(f"F1 (spans):               {format_pct(span_metrics.f1)}")


if __name__ == "__main__":
    main()

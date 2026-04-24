import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider


@dataclass
class TokenMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, y_true: list[bool], y_pred: list[bool]) -> None:
        for truth, pred in zip(y_true, y_pred):
            if pred and truth:
                self.tp += 1
            elif pred and not truth:
                self.fp += 1
            elif truth and not pred:
                self.fn += 1
            else:
                self.tn += 1

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
        description="Benchmark Presidio on ai4privacy/pii-masking-300k validation split."
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
        "--plot-path",
        default="benchmark_presidio_confusion_matrix.png",
        help="Dateipfad fuer den gespeicherten Confusion-Matrix-Plot.",
    )
    parser.add_argument(
        "--output-path",
        default="benchmark_presidio_results.txt",
        help="Dateipfad fuer den gespeicherten Text-Report der Metriken.",
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


def plot_confusion_matrix(metrics: TokenMetrics, output_path: str, title: str) -> None:
    counts = np.array(
        [
            [metrics.tp, metrics.fn],
            [metrics.fp, metrics.tn],
        ],
        dtype=float,
    )
    total = counts.sum() if counts.sum() else 1.0
    signed = np.array(
        [
            [counts[0, 0] / total, -counts[0, 1] / total],
            [-counts[1, 0] / total, counts[1, 1] / total],
        ]
    )
    max_abs = float(np.max(np.abs(signed))) if np.any(signed) else 1.0

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    heatmap = ax.imshow(
        signed,
        cmap="RdYlGn",
        vmin=-max_abs,
        vmax=max_abs,
        interpolation="nearest",
    )

    labels = np.array([["TP", "FN"], ["FP", "TN"]])
    for i in range(2):
        for j in range(2):
            count = int(counts[i, j])
            pct = counts[i, j] / total * 100.0
            ax.text(
                j,
                i,
                f"{labels[i, j]}\n{count:,}\n{pct:.2f}%",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="#0f172a",
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Vorhersage: PII", "Vorhersage: Kein PII"], fontsize=11, color="#0f172a")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Wahrheit: PII", "Wahrheit: Kein PII"], fontsize=11, color="#0f172a")
    ax.set_title(
        f"{title}\nGruen = korrekt, Rot = Fehler (False Positives/False Negatives)",
        fontsize=14,
        pad=16,
        color="#0f172a",
    )

    for edge in ax.spines.values():
        edge.set_color("#cbd5e1")
        edge.set_linewidth(1.2)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="#e2e8f0", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Signierte Token-Rate", color="#0f172a")
    cbar.ax.yaxis.set_tick_params(color="#0f172a")
    plt.setp(cbar.ax.get_yticklabels(), color="#0f172a")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    analyzer = build_analyzer()
    dataset = load_validation_dataset(args.dataset, args.split)

    if args.filter_language and "language" in dataset.column_names:
        lang = args.filter_language.strip().lower()
        dataset = dataset.filter(lambda row: str(row.get("language", "")).lower() == lang)
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

    report_lines = [
        "==== Presidio Benchmark (ai4privacy/pii-masking-300k) ====",
        f"Samples: {len(dataset)}",
        f"Scope Precision (tokens): {format_pct(token_metrics.precision)}",
        f"Recall (tokens):          {format_pct(token_metrics.recall)}",
        f"F1 (tokens):              {format_pct(token_metrics.f1)}",
        f"F1 (spans):               {format_pct(span_metrics.f1)}",
    ]
    plot_confusion_matrix(
        token_metrics,
        args.plot_path,
        "Presidio Token-Level Confusion Matrix",
    )
    report_lines.append(f"Confusion Matrix Plot:    {args.plot_path}")

    report = "\n".join(report_lines)
    Path(args.output_path).write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"Results file:             {args.output_path}")


if __name__ == "__main__":
    main()


import argparse
import ast
import re
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate token distribution (PII vs non-PII) on dataset subsets."
    )
    parser.add_argument("--dataset", default="ai4privacy/open-pii-masking-500k-ai4privacy")
    parser.add_argument("--split", default="validations")
    parser.add_argument(
        "--filter-language",
        default="de",
        help="Dataset language filter, e.g. de or en.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument(
        "--plot-path",
        default="dataset_pii_distribution_de_validations.png",
        help="Path for the saved plot.",
    )
    parser.add_argument(
        "--output-path",
        default="dataset_pii_distribution_de_validations.txt",
        help="Path for the saved text report.",
    )
    return parser.parse_args()


def load_validation_dataset(dataset_name: str, split_name: str):
    try:
        return load_dataset(dataset_name, split=split_name)
    except Exception:
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


def plot_distribution(pii_tokens: int, non_pii_tokens: int, output_path: str, title: str) -> None:
    total = max(1, pii_tokens + non_pii_tokens)
    pii_pct = pii_tokens / total * 100.0
    non_pii_pct = non_pii_tokens / total * 100.0

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    labels = ["Tokens"]
    ax.barh(labels, [non_pii_tokens], color="#94a3b8", label=f"Non-PII ({non_pii_pct:.2f}%)")
    ax.barh(labels, [pii_tokens], left=[non_pii_tokens], color="#ef4444", label=f"PII ({pii_pct:.2f}%)")

    ax.set_title(title, fontsize=14, color="#0f172a", pad=14)
    ax.set_xlabel("Token count", color="#0f172a")
    ax.tick_params(colors="#0f172a")
    ax.grid(axis="x", color="#e2e8f0", linestyle="-", linewidth=1)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")

    ax.text(
        non_pii_tokens / 2,
        0,
        f"{non_pii_tokens:,}",
        va="center",
        ha="center",
        color="#0f172a",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        non_pii_tokens + (pii_tokens / 2),
        0,
        f"{pii_tokens:,}",
        va="center",
        ha="center",
        color="#ffffff",
        fontsize=11,
        fontweight="bold",
    )

    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset = load_validation_dataset(args.dataset, args.split)

    if args.filter_language and "language" in dataset.column_names:
        lang = args.filter_language.strip().lower()
        dataset = dataset.filter(lambda row: str(row.get("language", "")).lower() == lang)

    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    total_tokens = 0
    pii_tokens = 0
    non_empty_samples = 0

    for row in tqdm(dataset, total=len(dataset), desc="Distribution", unit="sample", dynamic_ncols=True):
        text = row.get("source_text") or ""
        if not text:
            continue
        non_empty_samples += 1

        gt_spans = parse_span_labels(row.get("span_labels"))
        token_spans = extract_token_spans(text)
        y_true = mark_tokens_as_pii(token_spans, gt_spans)

        sample_total = len(y_true)
        sample_pii = int(np.sum(y_true))
        total_tokens += sample_total
        pii_tokens += sample_pii

    non_pii_tokens = total_tokens - pii_tokens
    pii_ratio = (pii_tokens / total_tokens) if total_tokens else 0.0

    title = f"PII Token Distribution ({args.filter_language or 'all'}, {args.split})"
    plot_distribution(pii_tokens, non_pii_tokens, args.plot_path, title)

    report_lines = [
        "==== Dataset PII Token Distribution ====",
        f"Dataset:                   {args.dataset}",
        f"Split:                     {args.split}",
        f"Language filter:           {args.filter_language or '(none)'}",
        f"Samples (rows):            {len(dataset)}",
        f"Samples (non-empty text):  {non_empty_samples}",
        f"Tokens total:              {total_tokens:,}",
        f"Tokens PII:                {pii_tokens:,}",
        f"Tokens non-PII:            {non_pii_tokens:,}",
        f"PII share:                 {pii_ratio * 100:.2f}%",
        f"Plot file:                 {args.plot_path}",
    ]

    report = "\n".join(report_lines)
    Path(args.output_path).write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"Results file:              {args.output_path}")


if __name__ == "__main__":
    main()

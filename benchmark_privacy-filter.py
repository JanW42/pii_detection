import argparse
import ast
import re
from dataclasses import dataclass
from typing import Iterable

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer


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
        description="Benchmark openai/privacy-filter on ai4privacy/pii-masking-300k validation split."
    )
    parser.add_argument("--model", default="openai/privacy-filter")
    parser.add_argument("--dataset", default="ai4privacy/pii-masking-300k")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument(
        "--filter-language",
        default="",
        help="Filter auf eine Datensatzsprache, z. B. english oder german.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--require-cu124",
        action="store_true",
        help="Bricht ab, wenn CUDA verfuegbar ist, aber nicht Runtime 12.4 (torch.version.cuda != 12.4).",
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


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def validate_cuda_runtime(require_cu124: bool) -> None:
    if not require_cu124:
        return
    runtime = torch.version.cuda
    if runtime != "12.4":
        raise RuntimeError(
            "CUDA Runtime 12.4 wurde angefordert (--require-cu124), "
            f"aber gefunden wurde: {runtime or 'keine CUDA Runtime'}."
        )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA wurde angefordert, ist aber nicht verfuegbar.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def label_is_pii(label: str) -> bool:
    if not label:
        return False
    normalized = label.upper()
    return normalized != "O"


def predict_spans_for_batch(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForTokenClassification,
    device: torch.device,
) -> list[list[tuple[int, int]]]:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
    )
    offset_mapping = encoded.pop("offset_mapping")
    inputs = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = logits.argmax(dim=-1).cpu()
    input_ids = encoded["input_ids"]

    batch_spans: list[list[tuple[int, int]]] = []
    for sample_idx in range(len(texts)):
        spans: list[tuple[int, int]] = []
        token_ids = input_ids[sample_idx]
        token_offsets = offset_mapping[sample_idx]
        sample_pred_ids = predicted_ids[sample_idx]

        for token_id, class_id, (start, end) in zip(token_ids, sample_pred_ids, token_offsets):
            token_id_int = int(token_id.item())
            if token_id_int in tokenizer.all_special_ids:
                continue

            start_i = int(start.item())
            end_i = int(end.item())
            if end_i <= start_i:
                continue

            label = model.config.id2label[int(class_id.item())]
            if label_is_pii(label):
                spans.append((start_i, end_i))

        batch_spans.append(merge_spans(spans))
    return batch_spans


def main() -> None:
    args = parse_args()
    dataset = load_validation_dataset(args.dataset, args.split)

    if args.filter_language and "language" in dataset.column_names:
        lang = args.filter_language.strip().lower()
        dataset = dataset.filter(lambda row: str(row.get("language", "")).lower() == lang)

    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    device = resolve_device(args.device)
    if device.type == "cuda":
        validate_cuda_runtime(args.require_cu124)

    # Kurzer Torch-Check mit dem explizit gewuenschten Label "divice".
    print(f"torch divice: {device}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device)
    model.eval()

    token_metrics = TokenMetrics()
    span_metrics = SpanMetrics()

    batch_size = max(1, args.batch_size)
    texts_batch: list[str] = []
    gt_spans_batch: list[list[tuple[int, int]]] = []

    def flush_batch() -> None:
        if not texts_batch:
            return

        pred_spans_batch = predict_spans_for_batch(texts_batch, tokenizer, model, device)
        for text, gt_spans, pred_spans in zip(texts_batch, gt_spans_batch, pred_spans_batch):
            token_spans = extract_token_spans(text)
            y_true = mark_tokens_as_pii(token_spans, gt_spans)
            y_pred = mark_tokens_as_pii(token_spans, pred_spans)
            token_metrics.update(y_true, y_pred)
            span_metrics.update(set(gt_spans), set(pred_spans))

        texts_batch.clear()
        gt_spans_batch.clear()

    for row in tqdm(dataset, total=len(dataset), desc="Benchmark Samples"):
        text = row.get("source_text") or ""
        if not text:
            continue

        gt_spans = parse_span_labels(row.get("span_labels"))
        texts_batch.append(text)
        gt_spans_batch.append(gt_spans)

        if len(texts_batch) >= batch_size:
            flush_batch()

    flush_batch()

    print("==== Privacy Filter Benchmark (ai4privacy/pii-masking-300k) ====")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA runtime (torch.version.cuda): {torch.version.cuda}")
    print(f"Samples: {len(dataset)}")
    print(f"Scope Precision (tokens): {format_pct(token_metrics.precision)}")
    print(f"Recall (tokens):          {format_pct(token_metrics.recall)}")
    print(f"F1 (tokens):              {format_pct(token_metrics.f1)}")
    print(f"F1 (spans):               {format_pct(span_metrics.f1)}")


if __name__ == "__main__":
    main()

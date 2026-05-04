import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inference-Speed-Test auf ai4privacy/open-pii-masking-500k-ai4privacy "
            "(validation, language=de) mit zufaelligen Samples."
        )
    )
    parser.add_argument("--model", default="openai/privacy-filter")
    parser.add_argument("--dataset", default="ai4privacy/open-pii-masking-500k-ai4privacy")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--language", default="de")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=0, help="0 = tokenizer default.")
    parser.add_argument("--threads", type=int, default=4, help="CPU Threads fuer PyTorch.")
    parser.add_argument("--output", default="inference_speed_report.txt")
    parser.add_argument("--latency-csv", default="inference_speed_latencies_ms.csv")
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


def get_text_from_row(row: dict) -> str:
    for key in ("source_text", "text", "raw_text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def percentile(values: np.ndarray, p: float) -> float:
    return float(np.percentile(values, p))


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))

    dataset = load_validation_dataset(args.dataset, args.split)
    if args.language and "language" in dataset.column_names:
        target = args.language.strip().lower()
        dataset = dataset.filter(lambda row: str(row.get("language", "")).lower() == target)

    if len(dataset) == 0:
        raise ValueError("Keine Datensaetze nach Sprachfilter gefunden.")

    sample_count = min(max(1, args.samples), len(dataset))
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), k=sample_count)
    sampled = dataset.select(indices)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model.eval()

    device = torch.device("cpu")
    model.to(device)

    warmup = min(max(0, args.warmup), sample_count - 1 if sample_count > 1 else 0)
    latencies_ms: list[float] = []
    token_lengths: list[int] = []

    total_start = time.perf_counter()
    for idx, row in enumerate(tqdm(sampled, total=sample_count, desc="Inference", unit="sample", dynamic_ncols=True)):
        text = get_text_from_row(row)
        if not text:
            continue

        tokenizer_kwargs = {"truncation": True, "return_tensors": "pt"}
        if args.max_length > 0:
            tokenizer_kwargs["max_length"] = args.max_length

        start = time.perf_counter()
        encoded = tokenizer(text, **tokenizer_kwargs)
        token_lengths.append(int(encoded["input_ids"].shape[1]))
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            _ = model(**encoded).logits
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if idx >= warmup:
            latencies_ms.append(elapsed_ms)

    total_elapsed_s = time.perf_counter() - total_start
    if not latencies_ms:
        raise RuntimeError("Keine gueltigen Latenzen gemessen.")

    lat_arr = np.array(latencies_ms, dtype=np.float64)
    tok_arr = np.array(token_lengths[warmup : warmup + len(latencies_ms)], dtype=np.float64)

    mean_ms = float(lat_arr.mean())
    p90_ms = percentile(lat_arr, 90)
    p95_ms = percentile(lat_arr, 95)
    median_ms = percentile(lat_arr, 50)
    throughput = len(latencies_ms) / total_elapsed_s if total_elapsed_s > 0 else 0.0

    report_lines = [
        "==== Inference Speed Report ====",
        f"Model:                    {args.model}",
        f"Dataset:                  {args.dataset}",
        f"Split / Language:         {args.split} / {args.language}",
        f"Sample target:            {sample_count} (seed={args.seed})",
        f"Warmup skipped:           {warmup}",
        f"Measured samples:         {len(latencies_ms)}",
        f"Avg token length:         {tok_arr.mean():.1f}" if len(tok_arr) else "Avg token length:         n/a",
        f"Mean latency:             {mean_ms:.2f} ms",
        f"Median latency:           {median_ms:.2f} ms",
        f"P90 latency:              {p90_ms:.2f} ms",
        f"P95 latency:              {p95_ms:.2f} ms",
        f"Throughput (wall clock):  {throughput:.2f} samples/s",
        f"Min / Max latency:        {lat_arr.min():.2f} / {lat_arr.max():.2f} ms",
        f"PyTorch threads:          {torch.get_num_threads()}",
    ]

    report = "\n".join(report_lines)
    Path(args.output).write_text(report + "\n", encoding="utf-8")

    with Path(args.latency_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "latency_ms", "token_length"])
        for i, (lat, tlen) in enumerate(zip(latencies_ms, tok_arr.tolist())):
            writer.writerow([i, f"{lat:.6f}", int(tlen)])

    print(report)
    print(f"Report file:              {args.output}")
    print(f"Latency CSV:              {args.latency_csv}")


if __name__ == "__main__":
    main()

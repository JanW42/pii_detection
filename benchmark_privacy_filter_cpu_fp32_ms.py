import argparse
import time

import numpy as np
import onnxruntime as ort
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizerFast


DATASET_NAME = "ai4privacy/open-pii-masking-500k-ai4privacy"
SPLIT_NAME = "validation"
LANGUAGE = "de"
MODEL_NAME = "openai/privacy-filter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU benchmark with selectable precision/quantization.")
    parser.add_argument(
        "--samples",
        default="1000",
        help="Number of samples to time, or 'all'.",
    )
    parser.add_argument(
        "--runtime",
        choices=["torch", "onnx"],
        default="torch",
        help="Inference runtime on CPU.",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16", "int8", "int4"],
        default="fp32",
        help="Inference mode on CPU.",
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


def load_tokenizer_robust(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        pass

    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    except Exception:
        pass

    tokenizer_file = hf_hub_download(repo_id=model_name, filename="tokenizer.json")
    return PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)


def build_model_for_precision(precision: str, device: torch.device):
    if precision == "fp32":
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32
        ).to(device)
        return model, "FP32"

    if precision == "fp16":
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float16
        ).to(device)
        return model, "FP16"

    if precision == "bf16":
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device)
        return model, "BF16"

    if precision == "int8":
        model_fp32 = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32
        ).to(device)
        model_fp32.eval()
        model_int8 = torch.quantization.quantize_dynamic(model_fp32, {torch.nn.Linear}, dtype=torch.qint8)
        return model_int8, "INT8 (dynamic quantization)"

    try:
        from torchao.quantization import Int4WeightOnlyConfig, quantize_  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "INT4 wurde angefordert, aber torchao mit Int4WeightOnlyConfig ist nicht verfuegbar. "
            "Bitte torchao installieren oder --precision int8 nutzen."
        ) from exc

    model_int4 = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)
    model_int4.eval()
    quantize_(model_int4, Int4WeightOnlyConfig())
    return model_int4, "INT4 (torchao weight-only)"


def build_onnx_session_for_precision(precision: str):
    if precision == "bf16":
        raise RuntimeError("BF16 ist fuer ONNX-Modellauswahl hier nicht vorgesehen. Nutze fp32/fp16/int8/int4.")

    filename_map = {
        "fp32": "model.onnx",
        "fp16": "model_fp16.onnx",
        "int8": "model_quantized.onnx",
        "int4": "model_q4.onnx",
    }
    model_filename = filename_map[precision]

    local_onnx_dir = snapshot_download(
        repo_id=MODEL_NAME,
        allow_patterns=[f"onnx/{model_filename}", f"onnx/{model_filename}_data*"],
    )
    model_path = f"{local_onnx_dir}/onnx/{model_filename}"

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = max(1, min(16, (torch.get_num_threads() or 1)))
    session = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
    return session, f"ONNX {precision.upper()} ({model_filename})"


def run_torch_forward(model, inputs):
    _ = model(**inputs).logits


def run_onnx_forward(session, inputs):
    ort_inputs = {
        "input_ids": inputs["input_ids"].cpu().numpy().astype(np.int64, copy=False),
        "attention_mask": inputs["attention_mask"].cpu().numpy().astype(np.int64, copy=False),
    }
    if "token_type_ids" in inputs:
        ort_inputs["token_type_ids"] = inputs["token_type_ids"].cpu().numpy().astype(np.int64, copy=False)
    _ = session.run(None, ort_inputs)


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    device = torch.device("cpu")

    dataset = load_validation_dataset(DATASET_NAME, SPLIT_NAME)
    if "language" not in dataset.column_names:
        raise ValueError("Spalte 'language' nicht im Datensatz gefunden.")

    dataset = dataset.filter(lambda row: str(row.get("language", "")).lower() == LANGUAGE)
    dataset = dataset.filter(lambda row: isinstance(row.get("source_text"), str) and len(row["source_text"].strip()) > 0)

    samples_arg = str(args.samples).strip().lower()
    if samples_arg == "all":
        sample_count = len(dataset)
    else:
        requested_samples = int(samples_arg)
        if requested_samples <= 0:
            raise ValueError("--samples muss eine positive Zahl oder 'all' sein.")
        sample_count = min(requested_samples, len(dataset))
    if sample_count == 0:
        raise ValueError("Keine passenden Samples gefunden.")
    dataset = dataset.select(range(sample_count))

    tokenizer = load_tokenizer_robust(MODEL_NAME)
    if args.runtime == "torch":
        model, precision_label = build_model_for_precision(args.precision, device)
        model.eval()
    else:
        session, precision_label = build_onnx_session_for_precision(args.precision)

    warmup_count = min(20, sample_count)
    for i in tqdm(range(warmup_count), desc="Warm-up", unit="sample", dynamic_ncols=True):
        text = dataset[i]["source_text"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=False)
        if args.runtime == "torch":
            run_torch_forward(model, inputs)
        else:
            run_onnx_forward(session, inputs)

    tokenizer_timings_ms: list[float] = []
    forward_timings_ms: list[float] = []
    total_timings_ms: list[float] = []
    for i in tqdm(range(sample_count), desc=f"Timing {precision_label}", unit="sample", dynamic_ncols=True):
        text = dataset[i]["source_text"]
        t0 = time.perf_counter()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=False)
        t1 = time.perf_counter()
        if args.runtime == "torch":
            run_torch_forward(model, inputs)
        else:
            run_onnx_forward(session, inputs)
        t2 = time.perf_counter()

        tokenizer_timings_ms.append((t1 - t0) * 1000.0)
        forward_timings_ms.append((t2 - t1) * 1000.0)
        total_timings_ms.append((t2 - t0) * 1000.0)

    avg_tokenizer_ms = sum(tokenizer_timings_ms) / len(tokenizer_timings_ms)
    avg_forward_ms = sum(forward_timings_ms) / len(forward_timings_ms)
    avg_total_ms = sum(total_timings_ms) / len(total_timings_ms)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Split: {SPLIT_NAME}")
    print(f"Language: {LANGUAGE}")
    print(f"Device: {device}")
    print(f"Runtime: {args.runtime}")
    print(f"Precision mode: {precision_label}")
    print(f"Samples timed: {len(total_timings_ms)}")
    print(f"Mean tokenizer time: {avg_tokenizer_ms:.3f} ms")
    print(f"Mean forward time:   {avg_forward_ms:.3f} ms")
    print(f"Mean total time:     {avg_total_ms:.3f} ms")


if __name__ == "__main__":
    main()

"""
python .\benchmark_privacy_filter_cpu_fp32_ms.py --precision fp32 --samples 1000
python .\benchmark_privacy_filter_cpu_fp32_ms.py --precision fp16 --samples 1000
python .\benchmark_privacy_filter_cpu_fp32_ms.py --precision bf16 --samples 1000
python .\benchmark_privacy_filter_cpu_fp32_ms.py --precision int8 --samples 1000
python .\benchmark_privacy_filter_cpu_fp32_ms.py --precision int4 --samples 1000

"""

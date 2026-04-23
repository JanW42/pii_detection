import threading
import time
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from asgiref.wsgi import WsgiToAsgi
from flask import Flask, jsonify, render_template, request
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

MODEL_NAME = "openai/privacy-filter"
ONNX_SUBFOLDER = "onnx"
MODEL_FILE = os.getenv("PRIVACY_FILTER_ONNX_FILE", "model_q4.onnx")
CPU_COUNT = os.cpu_count() or 4
DEFAULT_INTRA_THREADS = max(1, CPU_COUNT - 1)
INTRA_OP_THREADS = int(os.getenv("PRIVACY_FILTER_INTRA_OP_THREADS", str(DEFAULT_INTRA_THREADS)))
INTER_OP_THREADS = int(os.getenv("PRIVACY_FILTER_INTER_OP_THREADS", "1"))

app = Flask(__name__)

# Tokenizer and ONNX session are loaded once on startup for low request latency.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = INTRA_OP_THREADS
sess_options.inter_op_num_threads = INTER_OP_THREADS
sess_options.execution_mode = (
    ort.ExecutionMode.ORT_PARALLEL
    if INTER_OP_THREADS > 1
    else ort.ExecutionMode.ORT_SEQUENTIAL
)
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.enable_mem_pattern = True
sess_options.enable_cpu_mem_arena = True
sess_options.add_session_config_entry("session.intra_op.allow_spinning", "1")
sess_options.add_session_config_entry(
    "session.inter_op.allow_spinning",
    "1" if INTER_OP_THREADS > 1 else "0",
)
sess_options.add_session_config_entry("session.set_denormal_as_zero", "1")
id2label = {int(key): value for key, value in config.id2label.items()}
session: ort.InferenceSession | None = None
session_init_lock = threading.Lock()


def get_onnx_session() -> ort.InferenceSession:
    global session
    if session is not None:
        return session

    with session_init_lock:
        if session is not None:
            return session

        model_stem = Path(MODEL_FILE).stem
        model_dir = Path(
            snapshot_download(
                repo_id=MODEL_NAME,
                revision="main",
                allow_patterns=[
                    f"{ONNX_SUBFOLDER}/{MODEL_FILE}",
                    f"{ONNX_SUBFOLDER}/{model_stem}.onnx_data*",
                ],
            )
        )
        model_path = model_dir / ONNX_SUBFOLDER / MODEL_FILE
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX file not found: {model_path}. "
                f"Expected {MODEL_NAME}/{ONNX_SUBFOLDER}/{MODEL_FILE}"
            )

        session = ort.InferenceSession(
            str(model_path),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        return session

stats_lock = threading.Lock()
perf_stats = {
    "count": 0,
    "last_latency_ms": 0.0,
    "avg_latency_ms": 0.0,
    "total_latency_ms": 0.0,
}


def predict_entities(text: str) -> list[dict[str, str]]:
    onnx_session = get_onnx_session()
    encoded = tokenizer(text, return_tensors="np", return_offsets_mapping=True)
    offset_mapping = encoded["offset_mapping"][0].tolist()
    inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in encoded:
        inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

    logits = onnx_session.run(None, inputs)[0]
    predicted_ids = np.argmax(logits, axis=-1)[0]
    token_ids = encoded["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    result = []
    for token, class_id, (start, end) in zip(tokens, predicted_ids, offset_mapping):
        label = id2label.get(int(class_id), "O")
        if token in tokenizer.all_special_tokens:
            continue
        result.append(
            {
                "token": text[start:end] if end > start else token,
                "label": label,
                "start": start,
                "end": end,
            }
        )

    return result


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text") or ""

    if not text.strip():
        return jsonify({"error": "Bitte gib einen Text ein."}), 400

    start = time.perf_counter()
    try:
        predictions = predict_entities(text)
    except Exception as exc:
        return (
            jsonify(
                {
                    "error": (
                        "ONNX-Inferenz konnte nicht initialisiert werden. "
                        "Pruefe Modellzugriff und ONNX-Dateien im Repo "
                        f"{MODEL_NAME}/{ONNX_SUBFOLDER} ({MODEL_FILE}). "
                        f"Details: {exc}"
                    )
                }
            ),
            500,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    with stats_lock:
        perf_stats["count"] += 1
        perf_stats["last_latency_ms"] = latency_ms
        perf_stats["total_latency_ms"] += latency_ms
        perf_stats["avg_latency_ms"] = perf_stats["total_latency_ms"] / perf_stats["count"]
        snapshot = dict(perf_stats)

    return jsonify(
        {
            "input": text,
            "predictions": predictions,
            "perf": {
                "count": snapshot["count"],
                "last_latency_ms": round(snapshot["last_latency_ms"], 2),
                "avg_latency_ms": round(snapshot["avg_latency_ms"], 2),
            },
        }
    )


asgi_app = WsgiToAsgi(app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:asgi_app", host="127.0.0.1", port=8000, reload=True)

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import Response
from transformers import AutoConfig, AutoTokenizer


# ------------------------------
# Configuration via env vars
# ------------------------------
MODEL_NAME = os.getenv("PRIVACY_FILTER_MODEL_NAME", "openai/privacy-filter")
MODEL_PATH = Path(
    os.getenv(
        "PRIVACY_FILTER_ONNX_PATH",
        "varios-ai-filter/onnx/model_quantized.onnx",
    )
)
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
WORKERS = int(os.getenv("WORKERS", "4"))
CPU_COUNT = os.cpu_count() or 4
DEFAULT_INTRA_THREADS = max(2, CPU_COUNT // 2)
DEFAULT_INTER_THREADS = min(2, CPU_COUNT // 2)
INTRA_OP_THREADS = int(os.getenv("PRIVACY_FILTER_INTRA_OP_THREADS", str(DEFAULT_INTRA_THREADS)))
INTER_OP_THREADS = int(os.getenv("PRIVACY_FILTER_INTER_OP_THREADS", str(DEFAULT_INTER_THREADS)))
MAX_LENGTH = int(os.getenv("PRIVACY_FILTER_MAX_LENGTH", "512"))
STARTUP_RETRIES = int(os.getenv("PRIVACY_FILTER_STARTUP_RETRIES", "3"))
STARTUP_RETRY_DELAY_SEC = float(os.getenv("PRIVACY_FILTER_STARTUP_RETRY_DELAY_SEC", "1.5"))


# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)
logger = logging.getLogger("privacy_filter_service")


# ------------------------------
# API models
# ------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to analyze")


class EntityPrediction(BaseModel):
    token: str
    label: str
    start: int
    end: int


class PredictResponse(BaseModel):
    input: str
    predictions: list[EntityPrediction]
    latency_ms: float


# ------------------------------
# Runtime state
# ------------------------------
@dataclass
class RuntimeState:
    tokenizer: Any | None = None
    id2label: dict[int, str] | None = None
    session: ort.InferenceSession | None = None
    ready: bool = False
    init_error: str | None = None
    last_init_ts: float | None = None


state = RuntimeState()
state_lock = Lock()


# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(
    title="VARIOS AI Privacy Filter Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


def _build_session_options() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = INTRA_OP_THREADS
    opts.inter_op_num_threads = INTER_OP_THREADS
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
    opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
    opts.add_session_config_entry("session.set_denormal_as_zero", "1")
    return opts


def _log_onnx_runtime_config(session: ort.InferenceSession, opts: ort.SessionOptions) -> None:
    logger.info(
        (
            "ONNX Runtime config: providers=%s intra_op_threads=%d inter_op_threads=%d "
            "execution_mode=%s graph_optimization=%s mem_pattern=%s cpu_mem_arena=%s "
            "intra_op_spinning=%s inter_op_spinning=%s denormals_as_zero=%s"
        ),
        session.get_providers(),
        opts.intra_op_num_threads,
        opts.inter_op_num_threads,
        "ORT_PARALLEL" if opts.execution_mode == ort.ExecutionMode.ORT_PARALLEL else "ORT_SEQUENTIAL",
        str(opts.graph_optimization_level),
        str(opts.enable_mem_pattern),
        str(opts.enable_cpu_mem_arena),
        "1",
        "1",
        "1",
    )


def _load_runtime() -> None:
    with state_lock:
        if state.session is not None and state.tokenizer is not None and state.id2label is not None:
            state.ready = True
            return

        for attempt in range(1, STARTUP_RETRIES + 1):
            try:
                logger.info(
                    "Initializing runtime attempt=%d/%d model=%s onnx=%s",
                    attempt,
                    STARTUP_RETRIES,
                    MODEL_NAME,
                    MODEL_PATH,
                )
                if not MODEL_PATH.exists():
                    raise FileNotFoundError(
                        f"ONNX model not found at '{MODEL_PATH}'. "
                        "Expected model_quantized.onnx plus matching .onnx_data files in same folder."
                    )

                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
                config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
                id2label = {int(k): v for k, v in config.id2label.items()}

                sess_options = _build_session_options()
                session = ort.InferenceSession(
                    str(MODEL_PATH),
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )

                state.tokenizer = tokenizer
                state.id2label = id2label
                state.session = session
                state.ready = True
                state.init_error = None
                state.last_init_ts = time.time()

                logger.info("Runtime initialized with model=%s onnx=%s", MODEL_NAME, MODEL_PATH)
                _log_onnx_runtime_config(session, sess_options)
                return
            except Exception as exc:
                state.ready = False
                state.init_error = str(exc)
                logger.exception(
                    "Runtime initialization failed attempt=%d/%d",
                    attempt,
                    STARTUP_RETRIES,
                )
                if attempt < STARTUP_RETRIES:
                    time.sleep(STARTUP_RETRY_DELAY_SEC)

        raise RuntimeError(f"Runtime init failed after {STARTUP_RETRIES} attempts: {state.init_error}")


@app.on_event("startup")
def on_startup() -> None:
    logger.info("Service startup: host=%s port=%d workers=%d", HOST, PORT, WORKERS)
    try:
        _load_runtime()
    except Exception:
        logger.exception("Startup completed with degraded runtime state")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next: Any) -> Response:
    started = time.perf_counter()
    response: Response | None = None
    try:
        response = await call_next(request)
        return response
    finally:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        status_code = response.status_code if response is not None else 500
        logger.info(
            "request method=%s path=%s status=%d latency_ms=%.2f",
            request.method,
            request.url.path,
            status_code,
            latency_ms,
        )


@app.get("/health")
def health() -> dict[str, Any]:
    ready = (
        state.ready
        and state.session is not None
        and state.tokenizer is not None
        and state.id2label is not None
    )
    return {
        "status": "ok" if ready else "degraded",
        "model_name": MODEL_NAME,
        "onnx_path": str(MODEL_PATH),
        "runtime_ready": ready,
        "init_error": state.init_error,
        "last_init_ts": state.last_init_ts,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if state.session is None or state.tokenizer is None or state.id2label is None:
        try:
            _load_runtime()
        except Exception as exc:
            logger.exception("Prediction rejected due to unavailable runtime")
            raise HTTPException(status_code=503, detail=f"Runtime unavailable: {exc}") from exc

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")

    assert state.tokenizer is not None
    assert state.id2label is not None
    assert state.session is not None

    start = time.perf_counter()
    logger.debug("Prediction started text_len=%d", len(text))

    encoded = state.tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )
    offset_mapping = encoded.pop("offset_mapping")[0].tolist()

    inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in encoded:
        inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

    logits = state.session.run(None, inputs)[0]
    predicted_ids = np.argmax(logits, axis=-1)[0]
    token_ids = encoded["input_ids"][0]
    tokens = state.tokenizer.convert_ids_to_tokens(token_ids)

    predictions: list[EntityPrediction] = []
    for token, class_id, (start_idx, end_idx) in zip(tokens, predicted_ids, offset_mapping):
        if token in state.tokenizer.all_special_tokens:
            continue
        label = state.id2label.get(int(class_id), "O")
        predictions.append(
            EntityPrediction(
                token=text[start_idx:end_idx] if end_idx > start_idx else token,
                label=label,
                start=int(start_idx),
                end=int(end_idx),
            )
        )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "Prediction completed tokens=%d entities=%d latency_ms=%.2f",
        len(tokens),
        len(predictions),
        latency_ms,
    )
    return PredictResponse(input=text, predictions=predictions, latency_ms=latency_ms)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("privacy_filter_service:app", host=HOST, port=PORT, workers=WORKERS, reload=False)

# PII Detection Benchmark

Kompakter Benchmark zur PII-Erkennung auf `ai4privacy/pii-masking-300k`.

Im Fokus steht der Vergleich von zwei bidirektionalen Encoder-basierten Transformer-Ansaetzen:
- `openai/privacy-filter` (direkte Token-Klassifikation)
- Presidio-Pipeline (NLP-Encoder + regelbasierte Recognizer/Filter und Thresholding)

## Was wird gemessen?
- Token Precision
- Token Recall
- Token F1
- Span F1 (exakter Match)

## Setup (kurz)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional GPU (RTX 4070 Ti / CUDA 12.4):
```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## Benchmark ausfuehren
Privacy-Filter (alle Samples, deutsches Subset):
```powershell
python .\benchmark_privacy-filter.py --max-samples 0 --filter-language german
```

Mit erzwungenem CUDA-Check:
```powershell
python .\benchmark_privacy-filter.py --device cuda --require-cu124 --max-samples 0 --filter-language german
```

Presidio:
```powershell
python .\benchmark_presidio.py --max-samples 0 --filter-language german --language de
```

## Aktueller Snapshot
### Presidio (`ai4privacy/pii-masking-300k`, validation)
- Samples: `8120`
- Token Precision: `68.36%`
- Token Recall: `37.73%`
- Token F1: `48.62%`
- Span F1: `40.17%`

### `openai/privacy-filter` (`ai4privacy/pii-masking-300k`, validation)
- Device: `cuda`
- Torch: `2.6.0+cu124`
- CUDA Runtime: `12.4`
- Samples: `8120`
- Token Precision: `92.33%`
- Token Recall: `83.44%`
- Token F1: `87.66%`
- Span F1: `48.25%`

## Hinweis
`No matching distribution found for torch` bedeutet meist: Python-Version ist zu neu (z. B. `3.14`). Nutze `3.11` oder `3.12`.

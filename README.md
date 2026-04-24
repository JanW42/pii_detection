# PII Detection Benchmark

Kompakter Benchmark zur PII-Erkennung auf `ai4privacy/open-pii-masking-500k-ai4privacy`.

Im Fokus steht der Vergleich von zwei bidirektionalen Encoder-basierten Transformer-Ansaetzen:
- `openai/privacy-filter` (direkte Token-Klassifikation)
- Presidio-Pipeline (NLP-Encoder + regelbasierte Recognizer/Filter und Thresholding)

## Aktueller Snapshot - new State of the Art
### `openai/privacy-filter` (`ai4privacy/open-pii-masking-500k-ai4privacy`, validations)
- Device: `cuda`
- Torch: `2.6.0+cu124`
- CUDA Runtime: `12.4`
- Samples: `8120`
- Token Precision: `92.33%`
- Token Recall: `83.44%`
- Token F1: `87.66%`
- Span F1: `48.25%`

### Presidio (`ai4privacy/open-pii-masking-500k-ai4privacy`, validations)
- Samples: `8120`
- Token Precision: `68.36%`
- Token Recall: `37.73%`
- Token F1: `48.62%`
- Span F1: `40.17%`

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
Privacy-Filter (alle `de` Samples, inkl. Report + Plot):
```powershell
python .\benchmark_privacy-filter.py --dataset ai4privacy/open-pii-masking-500k-ai4privacy --split validations --filter-language de --max-samples 0 --output-path .\results_privacy_filter_de_all.txt --plot-path .\confusion_privacy_filter_de_all.png
```

Presidio (alle `de` Samples, inkl. Report + Plot):
```powershell
python .\benchmark_presidio.py --dataset ai4privacy/open-pii-masking-500k-ai4privacy --split validations --filter-language de --max-samples 0 --output-path .\results_presidio_de_all.txt --plot-path .\confusion_presidio_de_all.png
```

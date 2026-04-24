import argparse
import re
from pathlib import Path


PCT_RE = re.compile(r"^\s*([^:]+):\s*([0-9]+(?:[.,][0-9]+)?)%\s*$")
VAL_RE = re.compile(r"^\s*([^:]+):\s*(.+?)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse benchmark results.txt files and generate interpreted evaluation output."
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="Result files. If empty, the script auto-detects common benchmark result files.",
    )
    parser.add_argument(
        "--output-path",
        default="evaluation_results_interpreted.txt",
        help="Output file for interpreted evaluation.",
    )
    return parser.parse_args()


def auto_discover_inputs() -> list[Path]:
    patterns = [
        "results*.txt",
        "*benchmark*results*.txt",
        "benchmark_*_results.txt",
    ]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(Path(".").glob(pattern))
    deduped = sorted({path.resolve() for path in found if path.is_file()})
    return deduped


def parse_result_file(path: Path) -> dict:
    data: dict[str, object] = {
        "path": str(path),
        "title": path.stem,
        "metrics": {},
        "values": {},
    }

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("====") and line.endswith("===="):
            data["title"] = line.strip("=").strip()
            continue

        pct_match = PCT_RE.match(line)
        if pct_match:
            key = normalize_key(pct_match.group(1))
            val = float(pct_match.group(2).replace(",", "."))
            data["metrics"][key] = val
            continue

        val_match = VAL_RE.match(line)
        if val_match:
            key = normalize_key(val_match.group(1))
            val = val_match.group(2).strip()
            data["values"][key] = val

    return data


def normalize_key(key: str) -> str:
    k = key.lower()
    k = k.replace("(", " ").replace(")", " ")
    k = k.replace("-", " ").replace("/", " ")
    k = "_".join(part for part in re.split(r"\s+", k) if part)
    return k


def classify_score(score: float) -> str:
    if score >= 90:
        return "sehr stark"
    if score >= 80:
        return "stark"
    if score >= 70:
        return "solide"
    if score >= 60:
        return "ausbaufähig"
    return "schwach"


def metric_name(data: dict, preferred: list[str]) -> tuple[str | None, float | None]:
    metrics: dict = data["metrics"]
    for key in preferred:
        if key in metrics:
            return key, float(metrics[key])
    return None, None


def interpret_single(data: dict) -> list[str]:
    lines: list[str] = []
    title = str(data["title"])
    metrics: dict = data["metrics"]
    lines.append(f"- {title}")

    precision = metrics.get("scope_precision_tokens", metrics.get("precision_tokens"))
    recall = metrics.get("recall_tokens")
    f1_token = metrics.get("f1_tokens")
    f1_span = metrics.get("f1_spans")

    if f1_token is not None:
        lines.append(f"  Token-F1: {f1_token:.2f}% ({classify_score(float(f1_token))}).")
    if precision is not None and recall is not None:
        p = float(precision)
        r = float(recall)
        gap = p - r
        if gap > 8:
            lines.append("  Interpretation: eher konservativ (hohe Precision, niedrigere Recall).")
        elif gap < -8:
            lines.append("  Interpretation: eher aggressiv (hohe Recall, mehr False Positives).")
        else:
            lines.append("  Interpretation: ausgewogenes Precision/Recall-Verhältnis.")
    if f1_token is not None and f1_span is not None:
        diff = float(f1_token) - float(f1_span)
        if diff > 20:
            lines.append("  Span-Gap: groß, exakte Span-Grenzen sind ein Hauptproblem.")
        elif diff > 10:
            lines.append("  Span-Gap: moderat, Boundary-Fehler sind relevant.")
        else:
            lines.append("  Span-Gap: gering, Token- und Span-Qualität liegen nah beieinander.")

    if not metrics:
        lines.append("  Keine auswertbaren Prozentmetriken gefunden.")
    return lines


def compare_models(parsed: list[dict]) -> list[str]:
    lines: list[str] = []
    if len(parsed) < 2:
        return lines

    def best_for(metric_keys: list[str], label: str) -> str | None:
        scored: list[tuple[float, str]] = []
        for item in parsed:
            _, val = metric_name(item, metric_keys)
            if val is not None:
                scored.append((val, str(item["title"])))
        if not scored:
            return None
        val, title = max(scored, key=lambda x: x[0])
        return f"{label}: {title} ({val:.2f}%)"

    lines.append("Modellvergleich:")
    for text in [
        best_for(["f1_tokens"], "Bestes Token-F1"),
        best_for(["scope_precision_tokens", "precision_tokens"], "Beste Token-Precision"),
        best_for(["recall_tokens"], "Beste Token-Recall"),
        best_for(["f1_spans"], "Bestes Span-F1"),
    ]:
        if text:
            lines.append(f"- {text}")
    return lines


def main() -> None:
    args = parse_args()

    input_paths = [Path(p).resolve() for p in args.inputs] if args.inputs else auto_discover_inputs()
    if not input_paths:
        raise FileNotFoundError(
            "Keine Ergebnisdateien gefunden. Übergib Dateien mit --inputs, z. B. --inputs results_*.txt."
        )

    parsed = [parse_result_file(path) for path in input_paths if path.exists()]
    if not parsed:
        raise FileNotFoundError("Die angegebenen Ergebnisdateien konnten nicht gelesen werden.")

    lines: list[str] = []
    lines.append("==== Interpreted Benchmark Evaluation ====")
    lines.append(f"Dateien: {len(parsed)}")
    lines.append("")
    lines.append("Einzelauswertung:")
    for item in parsed:
        lines.extend(interpret_single(item))
    lines.append("")
    lines.extend(compare_models(parsed))

    output = "\n".join(lines) + "\n"
    Path(args.output_path).write_text(output, encoding="utf-8")
    print(output)
    print(f"Saved interpretation: {args.output_path}")


if __name__ == "__main__":
    main()

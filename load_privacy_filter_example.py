import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


def main() -> None:
    model_name = "openai/privacy-filter"
    text = "Mein Name ist Max Mustermann und meine E-Mail ist max@example.com."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)

    ner = pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0 if device == "cuda" else -1,
    )

    results = ner(text)
    print("\nDetected entities:")
    for r in results:
        print(
            f"- entity={r['entity_group']}, score={r['score']:.3f}, "
            f"span=({r['start']}, {r['end']}), text='{r['word']}'"
        )


if __name__ == "__main__":
    main()

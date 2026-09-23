import re

from transformers import PreTrainedTokenizerBase


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def normalize_english_for_darija(text: str) -> str:
    normalized = normalize_text(text)
    lowered = normalized.lower()

    # Remove do/does/did in direct and wh-questions because Darija often
    # does not use this auxiliary.
    lowered = re.sub(r"\b(what|where|when|why|how)\s+(do|does|did)\s+", r"\1 ", lowered)
    lowered = re.sub(r"\b(do|does|did)\s+(i|you|he|she|we|they)\s+", r"\2 ", lowered)
    lowered = normalize_text(lowered)

    if normalized and normalized[0].isupper():
        lowered = lowered[:1].upper() + lowered[1:]
    return lowered


def format_source(text: str, task_prefix: str) -> str:
    return f"{task_prefix}{normalize_english_for_darija(text)}"


def tokenize_example(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    task_prefix: str,
    max_input_len: int,
    max_target_len: int,
) -> dict:
    source_text = format_source(example["en"], task_prefix)
    target_text = normalize_text(example["darija"])

    model_inputs = tokenizer(
        source_text,
        max_length=max_input_len,
        truncation=True,
        padding=False,
    )

    labels = tokenizer(
        text_target=target_text,
        max_length=max_target_len,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

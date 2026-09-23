import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import FINAL_MODEL_DIR, TASK_PREFIX  # noqa: E402
from src.data.preprocess import normalize_english_for_darija  # noqa: E402


class DarijaTranslator:
    def __init__(self, model_path: str | Path = FINAL_MODEL_DIR):
        self.model_path = str(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(self.device)

    def translate(self, text: str, max_length: int = 64) -> str:
        normalized = normalize_english_for_darija(text)
        source = f"{TASK_PREFIX}{normalized.strip()}"
        inputs = self.tokenizer(source, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run translation inference")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=str(FINAL_MODEL_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    translator = DarijaTranslator(model_path=args.model_path)
    print(translator.translate(args.text))


if __name__ == "__main__":
    main()

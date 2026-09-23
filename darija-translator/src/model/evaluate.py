import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import DATASET_PATH, FINAL_MODEL_DIR, SEED  # noqa: E402
from src.data.load_data import load_parallel_data  # noqa: E402
from src.model.inference import DarijaTranslator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained Darija translator")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH))
    parser.add_argument("--model_path", type=str, default=str(FINAL_MODEL_DIR))
    parser.add_argument("--test_size", type=float, default=0.2)
    return parser.parse_args()


def simple_token_f1(pred: str, target: str) -> float:
    pred_tokens = pred.split()
    tgt_tokens = target.split()
    if not pred_tokens and not tgt_tokens:
        return 1.0
    if not pred_tokens or not tgt_tokens:
        return 0.0

    overlap = sum(1 for t in pred_tokens if t in tgt_tokens)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(tgt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    args = parse_args()
    df = load_parallel_data(args.dataset)
    _, test_df = train_test_split(df, test_size=args.test_size, random_state=SEED)

    translator = DarijaTranslator(args.model_path)
    preds = [translator.translate(text) for text in test_df["en"].tolist()]
    targets = test_df["darija"].tolist()

    exact_match = np.mean([p.strip() == t.strip() for p, t in zip(preds, targets)])
    avg_f1 = np.mean([simple_token_f1(p, t) for p, t in zip(preds, targets)])

    print(f"Evaluation samples: {len(test_df)}")
    print(f"Exact match: {exact_match:.3f}")
    print(f"Token-level F1: {avg_f1:.3f}")


if __name__ == "__main__":
    main()

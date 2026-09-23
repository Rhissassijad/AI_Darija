import argparse
import inspect
import sys
from pathlib import Path

from datasets import Dataset
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import (  # noqa: E402
    CHECKPOINT_DIR,
    DATASET_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    FINAL_MODEL_DIR,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    MODEL_NAME,
    SEED,
    TASK_PREFIX,
)
from src.data.augment import augment_parallel_dataset, save_augmented_preview  # noqa: E402
from src.data.load_data import load_parallel_data  # noqa: E402
from src.data.preprocess import tokenize_example  # noqa: E402
from src.utils.helpers import ensure_dir, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train English -> Darija translator")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH))
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--no_augment", action="store_true", help="Disable grammar-based synthetic augmentation")
    parser.add_argument(
        "--augmented_preview_path",
        type=str,
        default=str(ROOT / "data" / "processed" / "augmented_dataset.csv"),
        help="Where to save the combined (raw + synthetic) dataset preview",
    )
    return parser.parse_args()


def maybe_augment_data(df: DataFrame, args: argparse.Namespace) -> DataFrame:
    if args.no_augment:
        return df

    combined_df = augment_parallel_dataset(df)
    save_augmented_preview(combined_df, args.augmented_preview_path)
    print(f"Augmented dataset: {len(df)} -> {len(combined_df)} rows")
    print(f"Saved augmented preview to: {args.augmented_preview_path}")
    return combined_df


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    ensure_dir(CHECKPOINT_DIR)
    ensure_dir(FINAL_MODEL_DIR)

    df = load_parallel_data(args.dataset)
    df = maybe_augment_data(df, args)
    train_df, val_df = train_test_split(df, test_size=args.val_size, random_state=SEED)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

    preprocess_fn = lambda x: tokenize_example(  # noqa: E731
        x,
        tokenizer=tokenizer,
        task_prefix=TASK_PREFIX,
        max_input_len=MAX_INPUT_LENGTH,
        max_target_len=MAX_TARGET_LENGTH,
    )

    train_ds = train_ds.map(preprocess_fn, remove_columns=["en", "darija"])
    val_ds = val_ds.map(preprocess_fn, remove_columns=["en", "darija"])

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args_signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    strategy_key = "evaluation_strategy" if "evaluation_strategy" in args_signature.parameters else "eval_strategy"

    training_kwargs = {
        "output_dir": str(CHECKPOINT_DIR),
        strategy_key: "epoch",
        "save_strategy": "epoch",
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "logging_steps": 10,
        "predict_with_generate": True,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "report_to": "none",
        "seed": SEED,
    }
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": collator,
    }
    trainer_signature = inspect.signature(Seq2SeqTrainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))
    print(f"Saved final model to: {FINAL_MODEL_DIR}")


if __name__ == "__main__":
    main()

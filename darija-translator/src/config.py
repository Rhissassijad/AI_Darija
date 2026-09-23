from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
FINAL_MODEL_DIR = PROJECT_ROOT / "models" / "final_model"

MODEL_NAME = "google/mt5-small"
TASK_PREFIX = "translate English to Moroccan Darija: "

MAX_INPUT_LENGTH = 64
MAX_TARGET_LENGTH = 64
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 5e-5
SEED = 42

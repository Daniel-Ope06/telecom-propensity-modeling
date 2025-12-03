from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
MODELS_DIR: Path = PROJECT_ROOT / "models"

PROCESSED_DIR: Path = DATA_DIR / "processed"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"

WALLACE_RAW_PATH: Path = DATA_DIR / "wallacecommunications.csv"
WALLACE_CLEAN_PATH: Path = PROCESSED_DIR / "wallace_cleaned.csv"
TRAIN_DATA_PATH: Path = PROCESSED_DIR / "train.csv"
TEST_DATA_PATH: Path = PROCESSED_DIR / "test.csv"

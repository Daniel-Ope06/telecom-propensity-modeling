"""
0_prepare_data.py
-----------------
1. Loads raw data.
2. Cleans it (using src.data_cleaning).
3. Saves 'wallace_clean.csv' to 'data'
4. Splits it into Train (80%) and Test (20%).
5. Saves 'train.csv' and 'test.csv' to 'data'.

This ensures all models use the EXACT same data split.
"""
import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore

# autopep8: off
import sys
import pathlib
import os
# Add project root to path
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.data_cleaning import (  # noqa: E402
    load_wallace_data, clean_wallace_data
)
# autopep8: on

current_dir = os.path.dirname(os.path.abspath(__file__))

WALLACE_PATH: str = os.path.join(
    current_dir, "..", "data", "wallacecommunications.csv")
WALLACE_CLEAN_PATH: str = os.path.join(
    current_dir, "..", "data", "wallace_clean.csv")

TRAIN_PATH: str = os.path.join(current_dir, "..", "data", "train.csv")
TEST_PATH: str = os.path.join(current_dir, "..", "data", "test.csv")

wallace_raw: pd.DataFrame = load_wallace_data(WALLACE_PATH)
wallace_clean: pd.DataFrame = clean_wallace_data(wallace_raw)

wallace_clean.to_csv(WALLACE_CLEAN_PATH, index=False)

train_set, test_set = train_test_split(
    wallace_clean,
    test_size=0.2,
    random_state=42,
    stratify=wallace_clean['new_contract_this_campaign']
)

train_set.to_csv(TRAIN_PATH, index=False)
test_set.to_csv(TEST_PATH, index=False)

import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
WALLACE_PATH = os.path.join(
    current_dir, "..", "data", "wallacecommunications.csv")


def load_wallace_data(filepath: str = WALLACE_PATH) -> pd.DataFrame:
    """Loads raw data from a CSV file."""
    return pd.read_csv(filepath)

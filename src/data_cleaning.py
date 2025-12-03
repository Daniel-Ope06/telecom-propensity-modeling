import pandas as pd
from pandas import Series
from pathlib import Path


def load_wallace_data(filepath: Path) -> pd.DataFrame:
    """Loads raw data from a CSV file."""
    return pd.read_csv(filepath)


def clean_wallace_data(data: pd.DataFrame) -> pd.DataFrame:
    """Fix typos and inconsistencies."""
    clean_data: pd.DataFrame = data.copy()

    # -- Fix typos --
    # Fix 'n' -> 'no' in 'has_tv_package'
    clean_data['has_tv_package'] = clean_data['has_tv_package'].replace(
        {'n': 'no'}
    )

    # Fix 'cell' -> 'cellular' in 'last_contact'
    clean_data['last_contact'] = clean_data['last_contact'].replace(
        {'cell': 'cellular'}
    )

    # Drop the single row with typo 'j' in month
    clean_data = clean_data[
        clean_data['last_contact_this_campaign_month'] != 'j'
    ]

    # -- Handle 'days_since_last_contact' (-1 handling) --
    # Create new binary column True(1), False(0)
    clean_data['never_contacted'] = (
        clean_data['days_since_last_contact_previous_campaign'] == -1
    ).astype(int)

    # Replace -1 with 2x Max to push it to the far past
    contacted_mask: Series[bool] = clean_data[
        'days_since_last_contact_previous_campaign'] != -1
    max_days: int = clean_data.loc[
        contacted_mask, 'days_since_last_contact_previous_campaign'].max()
    fill_value: int = 2 * max_days
    clean_data['days_since_last_contact_previous_campaign'] = clean_data[
        'days_since_last_contact_previous_campaign'].replace(-1, fill_value)

    # -- Map Target to 0/1 --
    clean_data['new_contract_this_campaign'] = clean_data[
        'new_contract_this_campaign'].map({'yes': 1, 'no': 0})

    return clean_data

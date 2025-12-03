"""
Exploratory Data Analysis (EDA) Script.

USAGE:
    Run interactively in VS Code:
    Highlight the lines to run then "Shift+Enter" to view data and charts.
"""
# autopep8: off
import os
import matplotlib.pyplot as plt

# Needed to import from local modules
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.data_cleaning import (  # noqa: E402
    load_wallace_data,
    clean_wallace_data
)
# autopep8: on

current_dir = os.path.dirname(os.path.abspath(__file__))
WALLACE_PATH = os.path.join(
    current_dir, "..", "data", "wallacecommunications.csv")
WALLACE_CLEAN_PATH = os.path.join(
    current_dir, "..", "data", "wallace_clean.csv")

# Get the data
wallace = load_wallace_data(WALLACE_PATH)
wallace.head()
wallace.info()

# Viewing categorical columns
wallace["town"].value_counts()
wallace["country"].value_counts()
wallace["job"].value_counts()
wallace["married"].value_counts()
wallace["education"].value_counts()
wallace["arrears"].value_counts()
wallace["housing"].value_counts()

# Five records have 'n' instead of 'no' in 'has_tv_package'
wallace["has_tv_package"].value_counts()

# Two records have 'cell' instead of 'cellular' in 'last_contact'
wallace["last_contact"].value_counts()

# One record has 'j' instead of an actual month
wallace["last_contact_this_campaign_month"].value_counts()

wallace["outcome_previous_campaign"].value_counts()
wallace["new_contract_this_campaign"].value_counts()

# Summary of numerical attributes
wallace.describe()

# Plot histogram of each numerical attribute
wallace.hist(bins=50, figsize=(20, 15))
plt.show()

"""
OBSERVATIONS FROM HISTOGRAMS:

1. 'age':
    - Fairly normal distribution (bell curve) centered around 30-40.

2. 'current_balance':
    - Highly right-skewed.
    - Most customers have lower balances, but outliers exist.
    - ACTION: Requires scaling to handle outliers.

3. 'conn_tr':
    - Shows discrete bars (1, 2, 3, 4, 5) rather than a continuous curve.
    - This indicates it is categorical (connection type grouping ID).
    - ACTION: We will not treat it as a continuous number but as categorical.

4. 'days_since_last_contact_previous_campaign':
    - Huge spike at -1 (representing "Never Contacted").
    - This dominates the distribution.
    - ACTION:
        - Create a new binary column ('never_contacted') to capture this group.
        - Replace -1 with a value around double the max days.
"""

# Clean (Fix typos, handle -1, map target)
wallace = clean_wallace_data(wallace)
wallace.head()

# Save to CSV
wallace.to_csv(WALLACE_CLEAN_PATH, index=False)

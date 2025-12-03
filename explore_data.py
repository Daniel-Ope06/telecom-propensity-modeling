"""
Exploratory Data Analysis

USAGE:
    Run interactively in VS Code.
    Click 'Run Cell' or press Shift+Enter inside the cell blocks (# %%).
"""
# %%
import joblib  # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
from src.data_cleaning import clean_wallace_data
from src.config import (
    WALLACE_RAW_PATH, MODELS_DIR
)

# %%
# Get the data
wallace = pd.read_csv(WALLACE_RAW_PATH)
wallace.head()

# %%
# View info about each column
wallace.info()

# %%
# Viewing categorical columns
print(
    wallace["town"].value_counts(), "\n\n",
    wallace["country"].value_counts(), "\n\n",
    wallace["job"].value_counts(), "\n\n",
    wallace["married"].value_counts(), "\n\n",
    wallace["education"].value_counts(), "\n\n",
    wallace["arrears"].value_counts(), "\n\n",
    wallace["housing"].value_counts()
)

# %%
print(
    # Five records have 'n' instead of 'no' in 'has_tv_package'
    wallace["has_tv_package"].value_counts(),
    "\n\n",
    # Two records have 'cell' instead of 'cellular' in 'last_contact'
    wallace["last_contact"].value_counts(),
    "\n\n",
    # One record has 'j' instead of an actual month
    wallace["last_contact_this_campaign_month"].value_counts()
)

# %%
print(
    wallace["outcome_previous_campaign"].value_counts(), "\n\n",
    wallace["new_contract_this_campaign"].value_counts()
)

# %%
# Summary of numerical attributes
wallace.describe()

# %%
# Plot histogram of each numerical attribute
wallace.hist(bins=50, figsize=(20, 15))
plt.show()

# %%
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

# %%
# Clean (Fix typos, handle -1, map target)
wallace_clean = clean_wallace_data(wallace)
wallace_clean.head()

# %%
# --- NEW CUSTOMER TO TEST ---
new_customer_dict = {
    'ID': [12345678],
    'town': ['Bristol'],
    'country': ['UK'],
    'age': [35],
    'job': ['technician'],
    'married': ['single'],
    'education': ['secondary'],
    'arrears': ['no'],
    'current_balance': [50.0],
    'housing': ['yes'],
    'has_tv_package': ['no'],
    'last_contact': ['cellular'],
    'conn_tr': [2],
    'last_contact_this_campaign_day': [15],
    'last_contact_this_campaign_month': ['may'],
    'this_campaign': [1],
    'days_since_last_contact_previous_campaign': [2000],  # Manual fix for -1
    'contacted_during_previous_campaign': [0],
    'outcome_previous_campaign': ['unknown'],
    'never_contacted': [1]  # Manual flag addition
}
new_customer_df: pd.DataFrame = pd.DataFrame(new_customer_dict)
new_customer_df.head()

# %%
# --- MODEL 1: LOGISTIC REGRESSION
log_reg = joblib.load(MODELS_DIR / "logistic_regression.joblib")

# Probability of saying 'Yes' to new contract
prediction_prob = log_reg.predict_proba(new_customer_df)[:, 1][0]

# Get the Yes/No
prediction_label: str = log_reg.predict(new_customer_df)[0]

print(f"Propensity Score: {prediction_prob:.4f}")
print(f"Prediction: {
    'Yes to Contract' if prediction_label == 1 else 'No to Contract'
}")

# %%

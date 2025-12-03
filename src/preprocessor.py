from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.preprocessing import OneHotEncoder, RobustScaler  # type: ignore


def create_preprocessor():
    """
    Creates a Scikit-Learn ColumnTransformer.
    Includes Imputers for safety against future messy data.
    """

    categorical_columns = [
        'town', 'country', 'job', 'married', 'education',
        'arrears', 'housing', 'has_tv_package', 'last_contact', 'conn_tr',
        'last_contact_this_campaign_month', 'outcome_previous_campaign'
    ]

    numeric_columns = [
        'age', 'current_balance', 'this_campaign',
        'days_since_last_contact_previous_campaign',
        'contacted_during_previous_campaign',
        'last_contact_this_campaign_day', 'never_contacted'
    ]

    # OneHotEncoder handles the string categories
    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore', sparse_output=False)

    # RobustScaler is crucial for 'current_balance' outliers
    # This scaler is not influenced by a few large outliers.
    numeric_transformer = RobustScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ],
        # Drop columns not listed (ID and new_contract_this_campaign)
        remainder='drop'
    )

    return preprocessor

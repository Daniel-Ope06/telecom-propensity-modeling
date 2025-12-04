"""
train_logistic_regression.py
----------------------------
1. Trains a Logistic Regression model with Hyperparameter Tuning.
2. Optimizes for F1-Score to balance Precision (Cost) and Recall (Opportunity).
"""
import joblib  # type: ignore
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.metrics import (  # type: ignore
    classification_report, ConfusionMatrixDisplay, roc_auc_score
)
from src.preprocessor import create_preprocessor
from src.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH,
    MODELS_DIR, OUTPUT_DIR, FIGURES_DIR
)


def run() -> None:
    """Execute logistic regression model training"""
    print("\n--- Model 1: Logistic Regression ---\n")
    print("Loading data into model...")

    train_df: pd.DataFrame = pd.read_csv(TRAIN_DATA_PATH)
    test_df: pd.DataFrame = pd.read_csv(TEST_DATA_PATH)

    target: str = 'new_contract_this_campaign'

    X_train: pd.DataFrame = train_df.drop(columns=[target])
    y_train: Series = train_df[target]
    X_test: pd.DataFrame = test_df.drop(columns=[target])
    y_test: Series = test_df[target]

    pipeline: Pipeline = Pipeline(steps=[
        ('preprocessor', create_preprocessor()),
        ('classifier', LogisticRegression(
            # liblinear supports both L1 and L2 regularization.
            solver='liblinear',
            class_weight='balanced',
            random_state=0,
            max_iter=1000
        ))
    ])

    print("Tuning hyperparameter 'C' using GridSearch...")

    # Hyperparameter to tune: C
    param_grid: dict[str, list] = {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],  # values of C to test
        'classifier__penalty': ['l1', 'l2']
    }
    grid_search: GridSearchCV = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1',  # Because of imbalanced dataset (minority said 'Yes')
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # --- EVALUATE ---
    best_model: Pipeline = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # [:, 1] means "probability of Class 1 (Yes)"
    y_probs = best_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_probs)

    # --- SAVE MODEL ---
    model_path = MODELS_DIR / "logistic_regression.joblib"
    joblib.dump(best_model, model_path)

    # --- GENERATE & SAVE REPORT ---
    report_lines: list = [
        "--- LOGISTIC REGRESSION RESULTS ---",
        f"Best Parameters: {grid_search.best_params_}",
        f"Best CV Score (F1): {grid_search.best_score_:.4f}",
        f"Test Set AUC Score: {auc_score:.4f}",
        "",  # This empty string creates a blank line (double newline)
        "--- Test Set Classification Report ---",
        classification_report(y_test, y_pred, target_names=['No', 'Yes'])
    ]
    report_text: str = "\n".join(report_lines)

    report_path = OUTPUT_DIR / "logistic_regression_report.txt"
    with open(report_path, "w") as file:
        file.write(report_text)

    # --- GENERATE & SAVE FIGURE ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=['No', 'Yes'],
        cmap='Blues',
        normalize='true',
        ax=ax
    )
    plt.title(f"Logistic Regression (Best C={
        grid_search.best_params_['classifier__C']})")

    fig_path = FIGURES_DIR / "logistic_regression_cm.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    # --- FINAL USER FEEDBACK ---
    print("\n--- Training Model 1 Complete ---\n")
    print(f"Model saved to: {model_path}\n")
    print(f"Results saved to: {report_path}\n")
    print(f"Confusion Matrix saved to: {fig_path}\n")

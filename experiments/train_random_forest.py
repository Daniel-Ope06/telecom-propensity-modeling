"""
train_random_forest.py
----------------------
1. Performs GridSearch to tune model complexity (Depth vs. Trees).
2. Extracts feature importance to explain the 'Why' behind predictions.
"""
import joblib  # type: ignore
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier  # type: ignore
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
    """Execute random forest model training"""
    print("\n--- Model 2: Random Forest ---\n")
    print("Loading data into model...")

    train_df: pd.DataFrame = pd.read_csv(TRAIN_DATA_PATH)
    test_df: pd.DataFrame = pd.read_csv(TEST_DATA_PATH)

    target: str = 'new_contract_this_campaign'

    X_train: pd.DataFrame = train_df.drop(columns=[target])
    y_train: Series = train_df[target]
    X_test: pd.DataFrame = test_df.drop(columns=[target])
    y_test: Series = test_df[target]

    pipeline = Pipeline(steps=[
        ('preprocessor', create_preprocessor()),
        ('classifier', RandomForestClassifier(
            random_state=1, class_weight='balanced'
        ))
    ])

    print("Tuning hyperparameters (Trees, Depth, Leaves)...")

    # Hyperparameters
    # n_estimators: Stability (How many experts vote?)
    # max_depth: Complexity (How detailed are the rules?)
    # min_samples_leaf: Generalization (Prevent rules for single people)
    param_grid: dict[str, list] = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    grid_search: GridSearchCV = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
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
    model_path = MODELS_DIR / "random_forest.joblib"
    joblib.dump(best_model, model_path)

    # --- GENERATE & SAVE REPORT ---
    report_lines: list = [
        "--- RANDOM FOREST RESULTS ---",
        f"Best Parameters: {grid_search.best_params_}",
        f"Best CV Score (F1): {grid_search.best_score_:.4f}",
        f"Test Set AUC Score: {auc_score:.4f}",
        "",  # This empty string creates a blank line (double newline)
        "--- Test Set Classification Report ---",
        classification_report(y_test, y_pred, target_names=['No', 'Yes'])
    ]
    report_text: str = "\n".join(report_lines)

    report_path = OUTPUT_DIR / "random_forest_report.txt"
    with open(report_path, "w") as file:
        file.write(report_text)

    # --- GENERATE & SAVE FIGURES ---
    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=['No', 'Yes'],
        cmap='Greens',
        normalize='true',
        ax=ax
    )
    plt.title("Random Forest Confusion Matrix")
    fig_path = FIGURES_DIR / "random_forest_cm.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Feature Importance
    rf_model = best_model.named_steps['classifier']
    importances = rf_model.feature_importances_
    preprocessor = best_model.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out()

    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(
        feat_imp_df['Feature'],
        feat_imp_df['Importance'],
        color='forestgreen'
    )
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.xlabel("Importance Score")

    feat_fig_path = FIGURES_DIR / "random_forest_importance.png"
    plt.savefig(feat_fig_path, bbox_inches='tight', dpi=300)
    plt.close()

    # --- FINAL USER FEEDBACK ---
    print("\n--- Training Model 2 Complete ---\n")
    print(f"Model saved to: {model_path}\n")
    print(f"Results saved to: {report_path}\n")
    print(f"Confusion Matrix saved to: {fig_path}\n")
    print(f"Feature Importance saved to: {feat_fig_path}")

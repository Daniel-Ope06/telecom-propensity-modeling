"""
train_neural_network.py
-----------------------
1. Architecture (hidden_layer_sizes): Testing 'Wide' vs 'Deep' structures.
2. Activation: Testing 'relu' (modern) vs 'tanh' (smooth).
3. Regularization (alpha): Preventing overfitting in complex networks.
"""
import joblib  # type: ignore
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier  # type: ignore
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
    """Execute neural network model training"""
    print("\n--- Model 3: Neural Network (MLP) ---\n")
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
        ('classifier', MLPClassifier(
            random_state=2,
            max_iter=500,
            # Stops training if validation score stops improving
            early_stopping=True
        ))
    ])

    print("Tuning hyperparameters (Layers & Activation)...")

    # Hyperparameters
    param_grid: dict[str, list] = {
        'classifier__hidden_layer_sizes': [
            (50,),  # 1 Hidden Layer with 50 neurons
            (100,),  # 1 Wide Hidden Layer with 100 neurons
            (50, 25),  # 2 Hidden Layers (Deep & Narrow)
            (100, 50, 25)  # 3 Hidden Layers (Deep & Wide)
        ],

        'classifier__activation': [
            'tanh',  # smooth curve (-1 to 1)
            'relu'  # standard AI activation (0 to infinity)
        ],

        # L2 Regularization penalty
        # Higher alpha = simple model, Lower = complex model
        'classifier__alpha': [0.0001, 0.001, 0.01]
    }
    grid_search: GridSearchCV = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,  # to keep it fast
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
    model_path = MODELS_DIR / "neural_network.joblib"
    joblib.dump(best_model, model_path)

    # --- GENERATE & SAVE REPORT ---
    report_lines: list = [
        "--- NEURAL NETWORK RESULTS ---",
        f"Best Parameters: {grid_search.best_params_}",
        f"Best CV Score (F1): {grid_search.best_score_:.4f}",
        f"Test Set AUC Score: {auc_score:.4f}",
        "",  # This empty string creates a blank line (double newline)
        "--- Test Set Classification Report ---",
        classification_report(y_test, y_pred, target_names=['No', 'Yes'])
    ]
    report_text: str = "\n".join(report_lines)

    report_path = OUTPUT_DIR / "neural_network_report.txt"
    with open(report_path, "w") as file:
        file.write(report_text)

    # --- GENERATE & SAVE FIGURE ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=['No', 'Yes'],
        cmap='Purples',  # Purple for Neural Net (Convention)
        normalize='true',
        ax=ax
    )
    plt.title("Neural Network Confusion Matrix")
    fig_path = FIGURES_DIR / "neural_network_cm.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    # --- FINAL USER FEEDBACK ---
    print("\n--- Training Model 1 Complete ---\n")
    print(f"Model saved to: {model_path}\n")
    print(f"Results saved to: {report_path}\n")
    print(f"Confusion Matrix saved to: {fig_path}\n")

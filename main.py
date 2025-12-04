"""
Master Execution Script
Runs the entire Machine Learning Pipeline in order.
"""
from experiments import (
    prepare_data,
    train_logistic_regression,
    train_random_forest
)


def main():
    print("Starting Wallace Communications ML Pipeline...")

    # Prepare Data (Clean -> Split -> Save)
    prepare_data.run()

    # Model 1
    train_logistic_regression.run()

    # Model 2
    train_random_forest.run()


if __name__ == "__main__":
    main()

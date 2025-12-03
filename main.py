"""
Master Execution Script
Runs the entire Machine Learning Pipeline in order.
"""
from experiments import (
    prepare_data,
    train_logistic_regression
)


def main():
    print("Starting Wallace Communications ML Pipeline...")

    # Prepare Data (Clean -> Split -> Save)
    prepare_data.run()

    # Model 1
    train_logistic_regression.run()


if __name__ == "__main__":
    main()

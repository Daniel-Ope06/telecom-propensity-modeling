"""
Master Execution Script
Runs the entire Machine Learning Pipeline in order.
"""
from experiments import prepare_data


def main():
    print("Starting Wallace Communications ML Pipeline...")

    # Prepare Data (Clean -> Split -> Save)
    prepare_data.run()


if __name__ == "__main__":
    main()

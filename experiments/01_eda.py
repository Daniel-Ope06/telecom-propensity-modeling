"""
Exploratory Data Analysis (EDA) Script.

USAGE:
    Run interactively in VS Code:
    Highlight the lines to run then "Shift+Enter" to view data and charts.
"""
# autopep8: off
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.data_cleaning import (
    load_wallace_data
)
# autopep8: on

# Get the data
wallace = load_wallace_data()
wallace.head()
wallace.info()

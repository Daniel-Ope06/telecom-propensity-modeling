# 📡 Telecom Propensity Modeling

***Optimizing Call Center ROI***

> A Machine Learning pipeline to predict which landline customers will buy a mobile contract.

## 📋 Executive Summary

- **The Problem:** Wallace Communications spends heavily on call center campaigns with a low conversion rate of **$\approx$ 20%**.

- **The Solution:** A predictive model that identifies high-propensity customers, allowing the business to target only the top leads.

- **The Impact:** The deployed Random Forest model delivers a **64% Success Rate** (Precision). This should **triple the call center efficiency** while still capturing the majority (59%) of all market opportunities.

## 🛠️ Tools Used

- **Core:** Python, Pandas, NumPy

- **ML:** Scikit-Learn (Pipelines, GridSearchCV, ColumnTransformer)

- **Models:** Logistic Regression, Random Forest, Multi-Layer Perceptron (Neural Network)

## 📊 Results & Insights

The **Random Forest** was selected as the production model for its superior ROI balance.

| Model | Test AUC | Precision (Success Rate) | Recall (Capture Rate) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 0.7645 | 0.39 | 0.61 |
| **Random Forest** 🏆 | **0.8669** | **0.64** | **0.59** |
| **Neural Network** | 0.7818 | 0.66 | 0.38 |

## 📂 Project Structure

| Item | Description |
| :---  | :---  |
| `src/` | Cleaning, Preprocessing, Config |
| `experiments/` | Train, Tune, Evaluate |
| `data/` | Raw and Processed datasets |
| `models/` | Serialized *.joblib* models |
| `docs/` | Project documentation |
| `output/` | Model reports and Confusion Matrices |
| `explore_data.py` | EDA and visual insights |
| `main.py` | Entry point script to execute pipeline |

---

**Context:** Developed for the *Machine Learning* module at *University of Stirling*.

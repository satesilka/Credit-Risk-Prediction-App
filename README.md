# End-to-End Credit Risk Prediction App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_LINK_HERE)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project is a full-stack Machine Learning application designed to automate credit risk assessment for financial institutions. By analyzing demographic and financial data, the model predicts whether a loan applicant is a **"Good"** or **"Bad"** credit risk, helping to minimize capital loss and streamline the loan approval process.

The solution includes a production-ready **Random Forest** classification model optimized via `RandomizedSearchCV`, deployed as a real-time web application using **Streamlit**.



## Key Features

* **Real-Time Inference:** Instant credit assessments via an interactive web interface.
* **Optimized Performance:** Achieved **72% Accuracy** and **0.72 F1-Score** through extensive hyperparameter tuning and class balancing.
* **Feature Engineering:** Implemented log-transformations for skewed financial data (`Credit Amount`) and custom imputation strategies for missing account information.
* **Business Insights:** EDA revealed that applicants with 'little' or no checking account status have a **5.1% higher probability of default** compared to the average.

## Tech Stack

* **Language:** Python 3.9
* **Machine Learning:** Scikit-Learn (Random Forest, GridSearchCV), XGBoost
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit Cloud
* **Serialization:** Joblib

## Installation & Setup

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/satesilka/Credit-Risk-Prediction-App.git]
    (https://github.com/satesilka/Credit-Risk-Prediction-App)
    cd credit-risk-app
    ```

2.  **Create a virtual environment (Optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## Model Development Pipeline

The project follows a rigorous data science lifecycle:

1.  **Data Cleaning:** Handling missing values in `Saving accounts` and `Checking account` by treating them as a distinct "unknown" category (preserving risk signals).
2.  **EDA:** Identifying correlations between `Duration`, `Age`, and `Risk`.
3.  **Preprocessing:**
    * **Log Transformation:** Applied `np.log1p` to `Credit amount` to normalize skewed distribution.
    * **Encoding:** Custom Label Encoding for ordinal categorical variables.
4.  **Model Selection:** Benchmarked Decision Tree, Random Forest, Extra Trees, and XGBoost.
5.  **Optimization:** Used `RandomizedSearchCV` to tune `n_estimators`, `max_depth`, and `class_weight`.

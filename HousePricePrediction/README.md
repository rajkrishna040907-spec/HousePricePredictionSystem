# House Price Prediction System 🏡

An advanced Machine Learning regression project built during the AIML Summer Internship Capstone Project. This application predicts house prices based on structural and geographical characteristics, trained on exactly 2,000 real property records using Linear Regression, Random Forest, and XGBoost.

---

## 📁 Repository Structure

```text
House_Price_Prediction/
├── Dataset/             # Stores the raw 'house_prices.csv'
├── Notebook/            # Jupyter Notebooks for EDA and model training
├── Model/               # Saved serialized model files (.joblib)
├── Streamlit_App/       # Streamlit application code (app.py)
└── Documentation/       # Project Report and README.md
```

---

## ⚙️ Setup and Installation

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Install Dependencies
Run the following command to install the required libraries:
```bash
pip install pandas numpy scikit-learn xgboost seaborn streamlit joblib matplotlib
```

---

## 🚀 Execution Steps

### Step 1: Data Preprocessing & Model Training
To train the models and serialize them, run the training notebook located at `Notebook/house_price_prediction_analysis.ipynb`, or execute the quick training command from the project root:
```bash
python train_models.py
```
This script will:
- Clean and Winsorize outliers in the 2,000 records.
- Transform `YearBuilt` into `Property Age` (relative to 2026).
- Encode categories (`Location`, `Condition`, `Garage`).
- Train and save the baseline **Linear Regression**, **Random Forest**, and **XGBoost** models in `Model/`.

### Step 2: Launch the Streamlit Web App
Run the following command to start the web application:
```bash
streamlit run Streamlit_App/app.py
```
This will open the user interface in your default browser at `http://localhost:8501`.

---

## 📊 Model Evaluation Summary

All models were evaluated on an 80/20 train/test split of the 2,000-row dataset:

| Model | MAE ($) | RMSE ($) | R² Score |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | ~$9,200 | ~$11,400 | **0.9929** |
| **Random Forest** | ~$23,500 | ~$28,400 | **0.9618** |
| **XGBoost** | ~$14,200 | ~$17,400 | **0.9810** |

---

## 🛠️ Streamlit UI Features

The final application accepts structural and geographical features matching the dataset:
1. **Area (sq ft)**: Continuous size inputs.
2. **Bedrooms**: Number of bedrooms.
3. **Bathrooms**: Number of bathrooms.
4. **Total Floors**: Total levels in the property.
5. **Year Built**: Dynamic transformation to Property Age during inference (`2026 - YearBuilt`).
6. **Location**: Categorical selectbox (Rural, Suburban, Urban, Downtown).
7. **Property Condition**: Interactive slider (1 to 5 scale).
8. **Garage Available**: Binary toggle.

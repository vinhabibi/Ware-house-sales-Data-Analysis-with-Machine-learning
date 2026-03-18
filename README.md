# Warehouse Sales Data Analysis with Machine Learning

This project analyzes warehouse and retail sales patterns and builds machine learning models to predict retail sales.

## Project Overview

The notebook performs end-to-end analysis on the `Warehouse_and_Retail_Sales.csv` dataset:

- Data loading and initial inspection
- Data cleaning (missing values and duplicates)
- Exploratory data analysis (EDA)
- Feature engineering
- Data preprocessing for modeling
- Model training and evaluation
- Hyperparameter tuning for XGBoost

Main notebook:

- `Ware_house_sales_Data_Analysis_with_Machine_learning.ipynb`

## Project Structure

```text
Ware-house-sales-Data-Analysis-with-Machine-learning/
|- README.md
`- Ware_house_sales_Data_Analysis_with_Machine_learning.ipynb
```

## Dataset

Expected input file used in the notebook:

- `Warehouse_and_Retail_Sales.csv`

Core columns used include:

- `YEAR`, `MONTH`
- `SUPPLIER`, `ITEM TYPE`
- `RETAIL SALES`, `RETAIL TRANSFERS`, `WAREHOUSE SALES`
- `ITEM CODE`, `ITEM DESCRIPTION`

## Key Data Preparation Steps

1. Missing value handling
- Dropped rows with missing `RETAIL SALES` or `ITEM TYPE`
- Imputed missing `SUPPLIER` values with mode

2. Duplicate handling
- Removed duplicate rows

3. Feature engineering
- Applied `log1p` transformation to reduce skewness:
  - `RETAIL_SALES_LOG`
  - `RETAIL_TRANSFERS_LOG`
  - `WAREHOUSE_SALES_LOG`
- Built a `DATE` column from `YEAR` and `MONTH`
- Created time features:
  - `Quarter`, `DayOfWeek`, `IsWeekend`, `DayOfYear`, `WeekOfYear`

4. Encoding strategy
- One-hot encoding for low-cardinality `ITEM TYPE`
- Initially one-hot encoded `SUPPLIER`, then moved to target encoding to reduce dimensionality

5. Scaling
- Applied `StandardScaler` to selected numerical features

## Exploratory Data Analysis Highlights

- `WINE` is the dominant item type in the dataset
- Sales variables are strongly right-skewed with many outliers
- Strong positive correlation between `RETAIL SALES` and `RETAIL TRANSFERS` (~0.96)
- Seasonal behavior is visible in monthly sales trends
- A small set of suppliers contributes a large share of total sales

## Models Trained

- Random Forest Regressor
- XGBoost Regressor (with one-hot encoded features)
- XGBoost Regressor (with target encoded `SUPPLIER`)

Target variable:

- `RETAIL_SALES_LOG`

Train/test split:

- 80/20

## Results Summary

From the notebook outputs:

- Random Forest
  - R2: `0.9041`
  - RMSE: `0.3090`

- XGBoost
  - R2: `0.9145`
  - RMSE: `0.2917`

- XGBoost hyperparameter tuning (GridSearchCV)
  - Best parameters:
    - `learning_rate: 0.2`
    - `max_depth: 7`
    - `n_estimators: 200`
    - `subsample: 0.9`
  - Best cross-validation RMSE: `0.2908`

## How to Run

1. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost category-encoders
```

2. Place `Warehouse_and_Retail_Sales.csv` in the path expected by the notebook.

3. Open and run:

- `Ware_house_sales_Data_Analysis_with_Machine_learning.ipynb`

## Future Improvements

- Add lag and rolling time-series features
- Compare with LightGBM and CatBoost
- Improve outlier handling strategy
- Add model explainability (SHAP or feature importance dashboard)

## Author

Raymond (vinhabibi)

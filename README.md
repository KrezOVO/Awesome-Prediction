# Awesome Prediction Project

This project is designed for quick data preprocessing, outlier detection, and regression analysis using Python. It includes scripts for generating interactive 3D visualizations, performing various regression techniques (linear, ridge), and detecting outliers with DBSCAN, leveraging libraries like `pandas`, `plotly`, and `scikit-learn`.

## Repository Structure
- **data preprocessing/data.py**: 
  - Reads Excel data from multiple sheets.
  - Generates interactive 3D scatter plots using `plotly`.
  - Visualizes relationships between `Qv`, `DP`, `RPM`, and `M1`.

- **regression analysis/LinearRegression.py**: 
  - Performs linear regression on the data.
  - Uses `scikit-learn` to predict `M1` based on `Qv`, `DP`, and `RPM`.
  - Outputs model performance metrics (R², MSE) and saves results.

- **regression analysis/RidgeRegression.py**: 
  - Implements Ridge Regression (L2 regularization) to improve prediction by reducing overfitting.
  - Outputs R², MSE, and optimized Ridge model results.

- **outlier detection/dbscan.py**: 
  - Detects outliers in the dataset using the DBSCAN algorithm.
  - Clusters data points based on their density and marks outliers for further analysis.
  - Helps identify noisy data that could skew regression results.

## Dependencies
- Python 3.x
- pandas
- scikit-learn
- plotly
- openpyxl

Install dependencies via:
```bash
pip install -r requirements.txt
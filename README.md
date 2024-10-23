# Awesome Prediction Project

This project is designed for quick data preprocessing and linear regression analysis using Python. It includes scripts for generating interactive 3D visualizations and performing regression analysis on a dataset, leveraging libraries like `pandas`, `plotly`, and `scikit-learn`.

## Repository Structure
- **data preprocessing/data.py**: 
  - Reads Excel data from multiple sheets.
  - Generates interactive 3D scatter plots using `plotly`.
  - Visualizes relationships between `Qv`, `DP`, `RPM`, and `M1`.
  
- **regression analysis/LinearRegression.py**: 
  - Performs linear regression on the data.
  - Uses `scikit-learn` to predict `M1` based on `Qv`, `DP`, and `RPM`.
  - Outputs model performance metrics, including R² and MSE, and saves results.

## Dependencies
- Python 3.x
- pandas
- scikit-learn
- plotly
- openpyxl

Install dependencies via:
```bash
pip install requirements.txt

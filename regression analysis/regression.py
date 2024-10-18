import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    types = df['type'].unique().tolist()
    return {
        'types': types,
        'data': {t: df[df['type'] == t] for t in types}
    }

def perform_regression(data, t):
    X = data[['Qv', 'DP', 'RPM']]
    y = data['M1']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2

def print_results(sheet_name, t, model, mse, r2):
    print(f"表格: {sheet_name}, 类型: {t}")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"决定系数 (R^2): {r2:.4f}")
    print(f"模型系数 (Qv, DP, RPM): {model.coef_}")
    print(f"模型截距: {model.intercept_}")
    print("---")

def main():
    file_path = '1728528473034.xlsx'
    sheet_names = ['CVAF', 'CVAR', 'HFF', 'HDF']

    for sheet_name in sheet_names:
        sheet_data = load_data(file_path, sheet_name)

        for t in sheet_data['types']:
            data = sheet_data['data'][t]
            model, mse, r2 = perform_regression(data, t)
            print_results(sheet_name, t, model, mse, r2)

if __name__ == "__main__":
    main()

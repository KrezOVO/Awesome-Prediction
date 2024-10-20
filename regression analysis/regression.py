import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

def linear_regression(train_data, test_data):
    X_train = train_data[['Qv', 'DP', 'RPM']]
    y_train = train_data['M1']
    
    X_test = test_data[['Qv', 'DP', 'RPM']]
    y_test = test_data['M1']

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2

def print_results(mse, r2):
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"R^2 分数: {r2:.4f}")
    print("---")

def main():
    # 设置文件路径
    file_path = os.path.join('..', 'data_base.xlsx')

    # 定义sheet名称
    sheet_names = ['CVAF', 'CVAR', 'HFF', 'HDF']

    # 用户输入测试集的sheet编号
    print("可用的模式：")
    for i, sheet in enumerate(sheet_names, 1):
        print(f"{i}. {sheet}")
    sheet_index = int(input("请输入测试集的模式编号：")) - 1
    while sheet_index < 0 or sheet_index >= len(sheet_names):
        sheet_index = int(input("输入无效，请重新输入测试集的模式编号：")) - 1

    test_sheet = sheet_names[sheet_index]

    all_data = pd.DataFrame()
    test_data = None
    test_type = None

    # 读取选定sheet的数据
    df = pd.read_excel(file_path, sheet_name=test_sheet)
    types = df['type'].unique().tolist()

    # 用户输入测试集的type编号
    print(f"{test_sheet} 中可用的空调类型：")
    for i, t in enumerate(types, 1):
        print(f"{i}. {t}")
    type_index = int(input("请输入测试集的空调类型编号：")) - 1
    while type_index < 0 or type_index >= len(types):
        type_index = int(input("输入无效，请重新输入测试集的空调类型编号：")) - 1

    test_type = types[type_index]

    # 读取所有数据
    for sheet_name in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        sheet_types = df['type'].unique().tolist()
        for t in sheet_types:
            data = df[df['type'] == t]
            if sheet_name == test_sheet and t == test_type:
                test_data = data
            all_data = pd.concat([all_data, data])

    # 执行线性回归分析
    model, mse, r2 = linear_regression(all_data, test_data)

    # 打印结果
    print("训练集包含所有数据")
    print(f"测试集为 {test_sheet} 模式下, 空调型号为 {test_type} 的数据")
    print_results(mse, r2)

if __name__ == "__main__":
    main()

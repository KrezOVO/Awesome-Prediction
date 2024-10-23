import os
import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def kernel_ridge_regression(train_data, test_data):
    X_train = train_data[['Qv', 'DP', 'RPM']]
    y_train = train_data['M1']
    
    X_test = test_data[['Qv', 'DP', 'RPM']]
    y_test = test_data['M1']

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 创建并训练内核岭回归模型
    model = KernelRidge(alpha=1.0, kernel='rbf')
    model.fit(X_train_scaled, y_train)

    # 预测并计算指标
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2

def main():
    # 设置文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'data_base.xlsx')
    output_path = os.path.join(current_dir, 'RidgeRegression.xlsx')

    # 定义sheet名称
    sheet_names = ['CVAF', 'CVAR', 'HFF', 'HDF']

    # 读取所有数据
    all_data = {}
    train_data = pd.DataFrame()
    for sheet_name in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        types = df['type'].unique().tolist()
        all_data[sheet_name] = {t: df[df['type'] == t] for t in types}
        train_data = pd.concat([train_data, df])

    # 准备结果数据框
    results = []

    # 遍历每个sheet和type作为测试集
    for test_sheet in sheet_names:
        for test_type in all_data[test_sheet].keys():
            # 准备测试集
            test_data = all_data[test_sheet][test_type]

            # 执行内核岭回归分析
            mse, r2 = kernel_ridge_regression(train_data, test_data)

            # 存储结果
            results.append({
                'Sheet': test_sheet,
                'Type': test_type,
                'MSE': mse,
                'R2': r2
            })

    # 将结果转换为DataFrame并保存到Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_path, index=False)
    print(f"结果已保存到 {output_path}")

if __name__ == "__main__":
    main()

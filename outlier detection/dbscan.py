import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def load_data(file_path, sheet_names):
    #加载所有sheet的数据并合并
    all_data = []
    for sheet_name in sheet_names:
        # 读取单个sheet的数据
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # 选择需要的列
        selected_columns = ['Qv', 'DP', 'RPM', 'M1', 'type']
        data = df[selected_columns]
        
        # 添加sheet名称列
        data['sheet'] = sheet_name
        
        # 将数据添加到all_data列表中
        all_data.append(data)
    
    # 将所有数据合并成一个DataFrame
    return pd.concat(all_data, ignore_index=True)

def detect_outliers(data, eps=0.5, min_samples=5):
    #使用DBSCAN检测异常值
    # 选择数值列进行标准化
    numeric_columns = ['Qv', 'DP', 'RPM', 'M1']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_columns])

    # 使用DBSCAN进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_data)

    # 将聚类结果添加到原始数据中
    data['Cluster'] = clusters

    # 找出异常值（标记为-1的点）
    return data[data['Cluster'] == -1]

def main():
    # 设置文件路径
    file_path = os.path.join('..', 'data_base.xlsx')

    # 定义sheet名称
    sheet_names = ['CVAF', 'CVAR', 'HFF', 'HDF']

    # 加载并合并所有数据
    combined_data = load_data(file_path, sheet_names)

    # 检测异常值
    outliers = detect_outliers(combined_data)

    # 计算并打印异常值统计信息
    outlier_count = len(outliers)
    total_count = len(combined_data)
    outlier_percentage = (outlier_count / total_count) * 100

    print(f"检测到的异常值数量: {outlier_count}")
    print(f"异常值占总数据的百分比: {outlier_percentage:.2f}%")

    # 保存异常值到新的Excel文件
    outliers.to_excel('dbscan.xlsx', index=False)
    print("异常值数据已保存到 dbscan.xlsx 文件中")

if __name__ == "__main__":
    main()

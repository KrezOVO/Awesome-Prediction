import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from pyecharts.charts import Line,Scatter3D
from pyecharts import options as opts

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 配置 config
config_xAxis3D = "Qv"
config_yAxis3D = "DP"
config_zAxis3D = "RPM"
config_vAxis3D = "M1"

"""
数据结构按照
{ 
    "method"    :   ["CVAF","CVAF",...],
    "type"  :   ["T1X","T1X",...],
    "Qv"    :   [375,375,...],
    "DP"    :   [-138,-67,...],
    "RPM"   :   [2797,2553,...],
    "M1"    :   [57.03,55.88,...]
}
排布，一共1889条数据
"""

def load_data(dir = "../data"):
    all_sheets = pd.read_excel(os.path.join(dir, "data_after.xlsx"), sheet_name=None)
    dataFrame = dict()
    for sheet_name, df in all_sheets.items():
        print(f"Sheet name: {sheet_name}")
        for key in df.keys():
            for i in df[key]:
                dataFrame.setdefault(key,[]).append(i)
        for i in ([sheet_name] * len(df[key])):
            dataFrame.setdefault("method",[]).append(i) 
    # print(dataFrame.keys())
    # print(len(dataFrame["method"]))
    # print(len(dataFrame["type"]))
    # print(len(dataFrame["Qv"]))
    # print(len(dataFrame["DP"]))
    # print(len(dataFrame["RPM"]))
    # print(len(dataFrame["M1"]))
    return dataFrame
 
def plot_demo(): 
    # 数据
    data = [10, 20, 30, 40, 50, 60]
    
    # 创建折线图对象
    line = Line()
    
    # 添加数据
    line.add_xaxis(["A", "B", "C", "D", "E", "F"])
    line.add_yaxis("系列1", data)
    
    # 设置全局选项
    line.set_global_opts(title_opts=opts.TitleOpts(title="折线图示例"))
    
    # 渲染图表到文件
    line.render("line_chart.html")

def plot_debug(dataFrame, method='CVAF'):
    method_list = np.asarray(dataFrame["method"])
    type_list = np.asarray(dataFrame["type"])
    method_ind = np.where(method_list==method)[0].tolist()

    # 创建折线图对象
    line = Line()
    
    for type in set(dataFrame["type"]):
        type_ind = np.where(type_list==type)[0].tolist()
        inter_ind = [x for x in method_ind if x in type_ind]

        x_data = [dataFrame["Qv"][i] for i in inter_ind]
        y_data = [dataFrame["M1"][i] for i in inter_ind]

        # 添加数据
        line.add_xaxis(xaxis_data = x_data)
        line.add_yaxis(series_name = type, y_axis = y_data)

    # 设置全局选项
    line.set_global_opts(
        # title_opts=opts.TitleOpts(title="折线图堆叠"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        xaxis_opts=opts.AxisOpts(type_="value", boundary_gap=False))

        # 渲染图表到文件
    line.render(f"{method}_Qv_M1_data.html")

def plot_3D_debug(dataFrame, method='CVAF'):

    scatter3D = Scatter3D(init_opts=opts.InitOpts(width="1920px", height="1080px"))
    method_list = np.asarray(dataFrame["method"])
    type_list = np.asarray(dataFrame["type"])
    method_ind = np.where(method_list==method)[0].tolist()

    max_m1 = max(dataFrame[config_vAxis3D])
    min_m1 = min(dataFrame[config_vAxis3D])
    print(max_m1, min_m1)
    for type in set(dataFrame["type"]):
        type_ind = np.where(type_list==type)[0].tolist()
        inter_ind = [x for x in method_ind if x in type_ind]

        x_data = [dataFrame[config_xAxis3D][i] for i in inter_ind]
        y_data = [dataFrame[config_yAxis3D][i] for i in inter_ind]
        z_data = [dataFrame[config_zAxis3D][i] for i in inter_ind]
        v_data = [dataFrame[config_vAxis3D][i] for i in inter_ind]

        data = []
        for i in range(len(x_data)):
            data.append([x_data[i],y_data[i],z_data[i],v_data[i]])

        # 添加数据
        scatter3D.add(series_name = type,
                              data=data,
        xaxis3d_opts=opts.Axis3DOpts(
            name=config_xAxis3D,
            type_="value",
        ),
        yaxis3d_opts=opts.Axis3DOpts(
            name=config_yAxis3D,
            type_="value",
        ),
        zaxis3d_opts=opts.Axis3DOpts(
            name=config_zAxis3D,
            type_="value",
        ),
        grid3d_opts=opts.Grid3DOpts(width=100, height=100, depth=100),)
        scatter3D.set_global_opts(
        visualmap_opts=opts.VisualMapOpts(
                type_="color",
                is_calculable=True,
                dimension=3,
                pos_top="100",
                max_=max_m1,
                min_=min_m1,
                range_color=[
                    "#1710c0",
                    "#0b9df0",
                    "#00fea8",
                    "#00ff0d",
                    "#f5f811",
                    "#f09a09",
                    "#fe0300",
                ],
            )
        )
        scatter3D.render("scatter3d.html")

def dataPreprocess(dataFrame):
    QV_data = dataFrame[config_xAxis3D]
    DP_data = dataFrame[config_yAxis3D]
    RPM_data = dataFrame[config_zAxis3D]
    M1_data = dataFrame[config_vAxis3D]

    xdata = []
    for i in range(len(QV_data)):
        xdata.append([QV_data[i],DP_data[i],RPM_data[i]])

    X = np.array(xdata)
    M1_data = np.array(M1_data)

    X_train, X_test, y_train, y_test = train_test_split(X, M1_data, test_size=0.3)
    return X_train, X_test, y_train, y_test

def linearReg(X_train, X_test, y_train, y_test):
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    print("------------linearReg-----------")
    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

def bayesianRidgeReg(X_train, X_test, y_train, y_test):
    reg = linear_model.BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
    reg.set_params(alpha_init=1.0, lambda_init=1e-3)
    reg.fit(X_train, y_train)
    ymean, ystd = reg.predict(X_test, return_std=True)
    # print(ymean)
    print("------------bayesianRidgeReg-----------")
    print("std error: %.2f" % np.mean(ystd))

def ridgeReg(X_train, X_test, y_train, y_test):
    reg = linear_model.Ridge(alpha=1.0, random_state=0)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("------------ridgeReg-----------")
    # The coefficients
    print("Coefficients: \n", reg.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

def huberReg(X_train, X_test, y_train, y_test):
    reg = linear_model.HuberRegressor(alpha=0.0,epsilon=1)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("------------huberReg-----------")
    # The coefficients
    print("Coefficients: \n", reg.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

if __name__ == '__main__':
    df = load_data()
    # plot_debug(df)
    # plot_debug(df,'CVAR')
    # plot_debug(df,'HFF')
    # plot_debug(df,'HDF')
    # plot_3D_debug(df)
    # plot_demo()

    # 回归demo
    X_train, X_test, y_train, y_test = dataPreprocess(df)
    linearReg(X_train, X_test, y_train, y_test)
    bayesianRidgeReg(X_train, X_test, y_train, y_test)
    ridgeReg(X_train, X_test, y_train, y_test)
    huberReg(X_train, X_test, y_train, y_test)
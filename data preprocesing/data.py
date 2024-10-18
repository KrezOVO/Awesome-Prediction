import pandas as pd
import plotly.graph_objects as go
import webbrowser
import os

def create_3d_plot(sheet_data, sheet_name):
    # 定义从蓝色到红色的颜色范围
    colorscale = [
        [0, "rgb(0,0,255)"],
        [0.25, "rgb(0,255,255)"],
        [0.5, "rgb(0,255,0)"],
        [0.75, "rgb(255,255,0)"],
        [1, "rgb(255,0,0)"]
    ]

    fig = go.Figure()

    for t in sheet_data['types']:
        data = sheet_data['data'][t]
        
        scatter = go.Scatter3d(
            x=data['Qv'],
            y=data['DP'],
            z=data['RPM'],
            mode='markers',
            marker=dict(
                size=5,
                color=data['M1'],
                colorscale=colorscale,
                colorbar=dict(title='M1'),
                showscale=True
            ),
            text=[f'Type: {t}, M1: {m1}' for m1 in data['M1']],
            hoverinfo='text',
            name=t,
            visible=True  # 默认所有type都可见
        )
        
        fig.add_trace(scatter)

    # 更新布局
    fig.update_layout(
        scene=dict(
            xaxis_title='Qv',
            yaxis_title='DP',
            zaxis_title='RPM',
            aspectmode='cube'
        ),
        title=f'3D Visualization of {sheet_name}: Qv, DP, RPM, and M1',
        height=900,  # 增加高度
        width=1600,  # 增加宽度
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, t=40, b=0)  # 减小边距
    )

    return fig

def main():
    # 设置文件路径
    file_path = os.path.join('..', 'data_base.xlsx')

    # 定义sheet名称
    sheet_names = ['CVAF', 'CVAR', 'HFF', 'HDF']

    for sheet_name in sheet_names:
        # 读取数据
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        types = df['type'].unique().tolist()
        sheet_data = {
            'types': types,
            'data': {t: df[df['type'] == t] for t in types}
        }

        # 创建3D图
        fig = create_3d_plot(sheet_data, sheet_name)

        # 保存为HTML文件，使用相对路径
        output_path = f'3D_plot_{sheet_name}.html'
        fig.write_html(output_path, full_html=False, include_plotlyjs='cdn')

        # 自动打开生成的HTML文件
        webbrowser.open('file://' + os.path.abspath(output_path))

if __name__ == "__main__":
    main()

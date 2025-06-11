
import dash
import dash_core_components as dcc
import dash_html_components as html

# 初始化Dash应用
app = dash.Dash(__name__, title="雷达上位机", update_title=None)

# 全局变量，用于存储数据
globalBuffer = {
    "raw": {"x": [], "y": []},
}

# 设置应用布局
app.layout = html.Div(
    children=[
        # 实时更新的间隔设置
        dcc.Interval(id="interval-component", interval=1000 / 30, n_intervals=0),
        dcc.Graph(id="fig0"),
        dcc.Graph(id="fig1"),
        # 按钮
        html.Button("⏸暂停", id="btn-pause", n_clicks=0, style={"margin-right": "10px"}),
        html.Button("保存数据", id="btn-savedata", n_clicks=0),
        html.Div(id="output-state"),
    ],
)

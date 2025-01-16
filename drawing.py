import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sympy import bspline_basis

labels = ['NeRF-APT', 'NeRF2', 'MLP']
data1 = [29.06, 21.34, -8.81]
data2 = [27.63, 20.88, -8.89]
data3 = [24.38, 15.77, -11.38]



fig = make_subplots(rows=1, cols=3, subplot_titles=("Bedroom", "Conference", "Office"), horizontal_spacing=0.07)

# data1 = [2.73, 2.84, 26]
# data2 = [22.87, 20.87, -9.27]
# data3 = [0.866, 0.82, 0.11]

# fig = make_subplots(rows=1, cols=3, subplot_titles=("BLE-rssi", "MIMO-csi", "RFID-spectrum"), horizontal_spacing=0.09)

# Colors for each bar in the subplots
colors1 = ['#A9CCE3', '#A9DFBF', '#F5B7B1']
colors2 = ['tomato', '#008080', '#F5DEB3']
colors3 = ['pink', 'lime', 'cyan']

custom_ticks = [-14, -10] + list(range(-5, 31, 5))

# Add bars to the subplots
for i, data in enumerate(zip(data1, data2, data3)):
    fig.add_trace(go.Bar(x=[labels[i]], y=[data[0] + 14], base=-14, name=f'Data 1 - {labels[i]}', marker_color=colors2[i]), row=1, col=1)
    fig.add_trace(go.Bar(x=[labels[i]], y=[data[1] + 14], base=-14, name=f'Data 2 - {labels[i]}', marker_color=colors2[i]), row=1, col=2)
    fig.add_trace(go.Bar(x=[labels[i]], y=[data[2] + 14], base=-14, name=f'Data 3 - {labels[i]}', marker_color=colors2[i]), row=1, col=3)

# Automatic adjustment of y-axis based on the data
fig.update_yaxes(title_text="SNR(dB)", range=[-14, 30], row=1, col=1, automargin=True, title_standoff=0.1, tickmode='array',
    tickvals=custom_ticks,
    ticktext=[str(tick) for tick in custom_ticks])
fig.update_yaxes(title_text="SNR(dB)", range=[-14, 30], row=1, col=2, automargin=True, title_standoff=0.1, tickmode='array',
    tickvals=custom_ticks,
    ticktext=[str(tick) for tick in custom_ticks])
fig.update_yaxes(title_text="SNR(dB)", range=[-14, 30], row=1, col=3, automargin=True,title_standoff=0.1, tickmode='array',
    tickvals=custom_ticks,
    ticktext=[str(tick) for tick in custom_ticks])

for i, ann in enumerate(fig.layout.annotations):
    ann.update(y=ann.y + 0.05)  # 将每个标题向上移动

fig.update_yaxes(tickangle=45)
fig.update_xaxes(tickangle=45)

fig.update_yaxes()






fig.update_layout(
    title={
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
)
fig.update_layout(
    # 设置图表背景颜色
    plot_bgcolor='white',
    # 设置绘图区域背景颜色
    paper_bgcolor='white',
    # 添加图例（可选）
    showlegend=True
)
fig.update_layout(barmode='group', showlegend=False)
fig.write_image("synthetic.pdf")


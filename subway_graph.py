import sys
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QMessageBox
from PySide6.QtCore import Qt, QTimer
import random

class SubwayGraph:
    def __init__(self):
        self.nodes = pd.DataFrame(columns=['x', 'y', 'floor', 'type', 'capacity'])
        self.edges = pd.DataFrame(columns=['source', 'target', 'length', 'width', 'type'])

    def add_node(self, name, x, y, floor=1, node_type='corridor'):
        self.nodes.loc[name] = [x, y, floor, node_type, 0]

    def add_edge(self, source, target, length, width=2.0, edge_type='corridor'):
        new_edge = pd.DataFrame([{'source': source, 'target': target, 'length': length, 'width': width, 'type': edge_type}])
        self.edges = pd.concat([self.edges, new_edge], ignore_index=True)

    def load_data(self, nodes_path, edges_path):
        self.nodes = pd.read_csv(nodes_path, index_col=0)
        self.edges = pd.read_csv(edges_path)

class SubwayWindow(QMainWindow):
    def __init__(self, graph_data):
        super().__init__()
        self.graph = graph_data
        self.setWindowTitle('Subway Graph')
        self.resize(1000, 800)
        central_widget = QWidget()  # 设置主部件和布局
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        self.gl_widget = pg.GraphicsLayoutWidget()  # 初始化 PyQtGraph 画布
        self.gl_widget.setBackground('w')
        layout.addWidget(self.gl_widget)
        self.plot_item = self.gl_widget.addPlot(title="Station Map")
        self.plot_item.setAspectLocked(True)
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.draw_map()
        self.people_item = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0))
        self.plot_item.addItem(self.people_item)
        self.passengers = []
        for _ in range(10):
            planned_path = ['Entry_A', 'Security_1', 'Gate_In']  # 之后想想怎么生成这个吧。。
            start_node = planned_path[0]
            # 获取起点的坐标
            node_info = self.graph.nodes.loc[start_node]
            self.passengers.append({
                'id': _,
                'pos': np.array([node_info['x'], node_info['y']], dtype=float),
                'path': planned_path[1:],  # 剩下的路径
                'v': 0.1,
                'finished': False
            })
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(50)  # 每50ms刷新一次

    def update_simulation(self):
        # 这里写下一帧的逻辑, 遍历 self.passengers，更新他们的位置
        x_positions = []
        y_positions = []
        arrival_threshold = self.passengers[0]['v']
        for p in self.passengers:
            if p['finished']:  # 如果已经到达终点，停在原地
                x_positions.append(p['pos'][0])
                y_positions.append(p['pos'][1])
                continue
            if len(p['path']) == 0:
                p['finished'] = True
                continue
            next_node_name = p['path'][0]
            target_info = self.graph.nodes.loc[next_node_name]
            target_pos = np.array([target_info['x'], target_info['y']], dtype=float)
            direction_vector = target_pos - p['pos']
            distance = np.linalg.norm(direction_vector)
            if distance < arrival_threshold:
                p['pos'] = target_pos
                p['path'].pop(0)
            else:
                if distance > 0:
                    normalized_dir = direction_vector / distance
                else:
                    normalized_dir = np.zeros(2)
                p['pos'] += normalized_dir * p['v']
            x_positions.append(p['pos'][0])
            y_positions.append(p['pos'][1])
        # 刷新画布
        self.people_item.setData(x=x_positions, y=y_positions)

    def draw_map(self):
        edge_x = []
        edge_y = []
        coord_map = self.graph.nodes[['x', 'y']].T.to_dict('list')
        for _, row in self.graph.edges.iterrows():
            src_pos = coord_map[row['source']]
            dst_pos = coord_map[row['target']]
            edge_x.extend([src_pos[0], dst_pos[0]])
            edge_y.extend([src_pos[1], dst_pos[1]])
        self.lines_item = pg.PlotCurveItem(
            x=np.array(edge_x),
            y=np.array(edge_y),
            pen=pg.mkPen(color=(150, 150, 150), width=3),
            connect='pairs'
        )
        self.plot_item.addItem(self.lines_item)
        node_x = self.graph.nodes['x'].values
        node_y = self.graph.nodes['y'].values
        color_map = {
            'entry': (0, 255, 0),  # 绿
            'security': (255, 0, 0),  # 红
            'gate': (255, 165, 0),  # 橙
            'platform': (0, 0, 255),  # 蓝
            'corridor': (200, 200, 200)  # 灰
        }
        brushes = [pg.mkBrush(color_map.get(t, (0, 0, 0))) for t in self.graph.nodes['type']]
        self.nodes_item = pg.ScatterPlotItem(
            x=node_x,
            y=node_y,
            size=15,
            brush=brushes,
            pen=pg.mkPen(None),  # 无边框
            symbol='o',
            data=list(self.graph.nodes.index),
        )
        self.nodes_item.sigClicked.connect(self.on_node_clicked)
        self.plot_item.addItem(self.nodes_item)
        for name, row in self.graph.nodes.iterrows():
            text = pg.TextItem(text=name, color='k', anchor=(0.5, 0))
            text.setPos(row['x'], row['y'] + 1)
            self.plot_item.addItem(text)
        # 锁定视角
        min_x, max_x = self.graph.nodes['x'].min(), self.graph.nodes['x'].max()
        min_y, max_y = self.graph.nodes['y'].min(), self.graph.nodes['y'].max()
        padding = 10
        self.plot_item.setRange(
            xRange=(min_x - padding, max_x + padding),
            yRange=(min_y - padding, max_y + padding)
        )
        self.plot_item.disableAutoRange()

    def on_node_clicked(self, plot, points):
        point = points[0]
        pos = point.pos()
        info_text = f"坐标位置:\nX: {pos.x():.1f}\nY: {pos.y():.1f}"
        if hasattr(point, 'data') and point.data() is not None:
            info_text = f"节点名称: {point.data()}\n{info_text}"
        QMessageBox.information(self, "节点信息", info_text)

if __name__ == "__main__":
    sim = SubwayGraph()
    sim.load_data(r'data\nodes.csv', r'data\edges.csv')
    app = QApplication(sys.argv)
    window = SubwayWindow(sim)
    window.show()
    sys.exit(app.exec())

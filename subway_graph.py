import sys
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PySide6.QtCore import Qt

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
        self.plot_item = self.gl_widget.addPlot(title="Station Map (Floor 1)")
        self.plot_item.setAspectLocked(True)
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.draw_map()

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
            symbol='o'
        )
        self.nodes_item.sigClicked.connect(self.on_node_clicked)
        self.plot_item.addItem(self.nodes_item)
        for name, row in self.graph.nodes.iterrows():
            text = pg.TextItem(text=name, color='k', anchor=(0.5, 0))
            text.setPos(row['x'], row['y'] + 1)
            self.plot_item.addItem(text)

    def on_node_clicked(self, plot, points):
        print(f"Clicked point: {points[0].pos()}")

if __name__ == "__main__":
    sim = SubwayGraph()
    sim.load_data(r'data\nodes.csv', r'data\edges.csv')
    # 2. 启动应用
    app = QApplication(sys.argv)
    window = SubwayWindow(sim)
    window.show()
    sys.exit(app.exec())

import sys
import numpy as np
import networkx as nx
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QMessageBox
from PySide6.QtCore import Qt, QTimer
import random
import heapq
import math

class SubwayGraph:
    def __init__(self):
        self.G = nx.DiGraph()
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
        for node_name, row in self.nodes.iterrows():
            self.G.add_node(node_name, pos=(row['x'], row['y']))
        for _, row in self.edges.iterrows():
            self.G.add_edge(row['source'], row['target'], weight=row['length'])

    def get_nodes_by_type(self, node_type):
        return self.nodes[self.nodes['type'] == node_type].index.tolist()

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
            all_entries = self.graph.get_nodes_by_type('entry')
            all_platforms = self.graph.get_nodes_by_type('platform')
            start_node = random.choice(all_entries)  # string
            final_destination = random.choice(all_platforms)  # string
            # 获取起点的坐标
            node_info = self.graph.nodes.loc[start_node]
            planned_path = self.plan_path(start_node, final_destination)
            self.passengers.append({
                'id': _,
                'pos': np.array([node_info['x'], node_info['y']], dtype=float),
                'path': planned_path[1:],  # 剩下的路径
                'v': 0.5,
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
        for idx, row in self.graph.edges.iterrows():
            src_pos = coord_map[row['source']]
            dst_pos = coord_map[row['target']]
            edge_x.extend([src_pos[0], dst_pos[0]])
            edge_y.extend([src_pos[1], dst_pos[1]])
            line_item = pg.PlotCurveItem(
                x=np.array([src_pos[0], dst_pos[0]]),
                y=np.array([src_pos[1], dst_pos[1]]),
                pen=pg.mkPen(color=(150, 150, 150), width=5),
                clickable=True
            )
            line_item.edge_data = row
            line_item.sigClicked.connect(self.on_edge_clicked)
            self.plot_item.addItem(line_item)
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

    def on_edge_clicked(self, item, event):
        if hasattr(item, 'edge_data'):
            data = item.edge_data
            info_text = (f"起点: {data['source']}\n"
                         f"终点: {data['target']}\n"
                         f"长度: {data['length']}\n"
                         f"类型: {data['type']}")
            QMessageBox.information(self, "边信息", info_text)

    def plan_path(self, start, end):
        path = self.astar(start, end)
        return path

    def cal_dist(self, u_name, v_name):
        x1 = self.graph.nodes.loc[u_name]['x']
        y1 = self.graph.nodes.loc[u_name]['y']
        x2 = self.graph.nodes.loc[v_name]['x']
        y2 = self.graph.nodes.loc[v_name]['y']
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    def get_travel_cost(self, u, v, edge_data):
        base_cost = edge_data['weight']
        # TODO: 增加人流密度代价
        return base_cost
    def node_passable_check(self, node_name):
        # TODO: 增加节点容量检查
        return True

    def astar(self, start, target):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {node: float('inf') for node in self.graph.G.nodes}  # 从起点到n的已知最小代价
        g_score[start] = 0
        f_score = {node: float('inf') for node in self.graph.G.nodes}
        f_score[start] = self.cal_dist(start, target)  # f_score[n] = g_score[n] + h(n)
        open_set_hash = {start}
        while open_set:
            current_f, current = heapq.heappop(open_set)
            open_set_hash.discard(current)
            if current == target:
                return self.reconstruct_path(came_from, current)
            for neighbor, edge_attr in self.graph.G[current].items():
                if not self.node_passable_check(neighbor):
                    continue
                cost = self.get_travel_cost(current, neighbor, edge_attr)
                test_g_score = g_score[current] + cost
                if test_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = test_g_score
                    f_score[neighbor] = g_score[neighbor] + self.cal_dist(neighbor, target)
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        return []

    def reconstruct_path(self,came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

if __name__ == "__main__":
    sim = SubwayGraph()
    sim.load_data(r'data\nodes.csv', r'data\edges.csv')
    app = QApplication(sys.argv)
    window = SubwayWindow(sim)
    window.show()
    sys.exit(app.exec())

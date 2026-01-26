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
        self.nodes['curr_capacity'] = 0
        self.edges['curr_capacity'] = 0
        self.edge_lookup = {}
        for idx, row in self.edges.iterrows():
            self.edge_lookup[(row['source'], row['target'])] = idx

    def get_nodes_by_type(self, node_type):
        return self.nodes[self.nodes['type'] == node_type].index.tolist()

sim = SubwayGraph()

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
        self.people_item = pg.ScatterPlotItem(size=3.5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 242, 0))
        self.plot_item.addItem(self.people_item)
        self.passengers = []
        for _ in range(100):
            self.passengers.append(Person(basic_v=(np.random.rand()*0.5+0.5)*0.5))
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(100)  # 刷新时间

    def update_simulation(self):
        # 这里写下一帧的逻辑, 遍历 self.passengers，更新他们的位置
        x_positions = []
        y_positions = []
        curr_passengers = []
        for p in reversed(self.passengers):
            px, py = p.update(p.v)
            x_positions.append(px)
            y_positions.append(py)
            if not p.finished:
                curr_passengers.append(p)
        self.passengers = curr_passengers
        self.people_item.setData(x=x_positions, y=y_positions)  # 刷新画布

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
                pen=pg.mkPen(color=(150, 150, 150), width=12),
                clickable=True
            )
            line_item.edge_index = idx
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
            size=18,
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
            info_text = f"{info_text}\n 当前人数: {self.graph.nodes.loc[point.data()]['curr_capacity']}"
        QMessageBox.information(self, "节点信息", info_text)

    def on_edge_clicked(self, item, event):
        if hasattr(item, 'edge_index'):
            idx = item.edge_index
            data = self.graph.edges.loc[idx]
            info_text = (f"起点: {data['source']}\n"
                         f"终点: {data['target']}\n"
                         f"长度: {data['length']}\n"
                         f"类型: {data['type']}\n"
                         f"当前人数: {data['curr_capacity']}\n"
                         f"人流密度：{data['curr_capacity']/(data['length']*data['width']):.2f}")
            QMessageBox.information(self, "边信息", info_text)

def get_travel_cost(graph: SubwayGraph, u, v, edge_data):
    #edge_idx = graph.edge_lookup[(u, v)]  # 但uv间有多条边咋办?算了先不考虑这个。。
    #base_cost = graph.edges.at[edge_idx, 'length']
    base_cost = edge_data['weight']
    # TODO: 增加人流密度代价
    intensity_cost = np.exp(graph.nodes.loc[v]['curr_capacity'] / graph.nodes.loc[v]['capacity']) - np.exp(graph.nodes.loc[u]['curr_capacity'] / graph.nodes.loc[u]['capacity'])  # 奖励去更稀疏的地方?
    intensity_cost /= 5
    return max(base_cost + intensity_cost, 0)

def node_passable_check(graph: SubwayGraph, node_name):
    return graph.nodes.loc[node_name]['capacity'] >= graph.nodes.loc[node_name]['curr_capacity']

def cal_dist(graph: SubwayGraph, u_name, v_name):
    x1 = graph.nodes.loc[u_name]['x']
    y1 = graph.nodes.loc[u_name]['y']
    x2 = graph.nodes.loc[v_name]['x']
    y2 = graph.nodes.loc[v_name]['y']
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def astar(graph: SubwayGraph, start, target):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph.G.nodes}  # 从起点到n的已知最小代价
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph.G.nodes}
    f_score[start] = cal_dist(graph, start, target)  # f_score[n] = g_score[n] + h(n)
    open_set_hash = {start}
    while open_set:
        current_f, current = heapq.heappop(open_set)
        open_set_hash.discard(current)
        if current == target:
            return reconstruct_path(came_from, current)
        for neighbor, edge_attr in graph.G[current].items():
            if not node_passable_check(graph, neighbor):
                continue
            cost = get_travel_cost(graph, current, neighbor, edge_attr)
            test_g_score = g_score[current] + cost
            if test_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = test_g_score
                f_score[neighbor] = g_score[neighbor] + cal_dist(graph, neighbor, target)
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)
    return []

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

def plan_path(graph: SubwayGraph, start, end):
    path = astar(graph, start, end)
    return path

class Person:
    def __init__(self, pid=None, basic_v=0.5, finished=False):
        self.id = pid
        self.basic_v = basic_v
        self.v = basic_v
        self.finished = finished
        all_entries = sim.get_nodes_by_type('entry')
        all_platforms = sim.get_nodes_by_type('platform')
        start_node = random.choice(all_entries)
        final_destination = random.choice(all_platforms)
        self.begin = start_node
        self.target = final_destination
        self.location = start_node
        self.x = sim.nodes.loc[start_node]['x']
        self.y = sim.nodes.loc[start_node]['y']
        self.path = plan_path(sim, self.location, self.target)[1:]
        self.in_node = True
        sim.nodes.at[self.location, 'curr_capacity'] += 1
        self.current_edge_idx = None

    def update(self, threshold=None):
        if not threshold:
            threshold = self.v
        if self.finished:
            return self.x, self.y
        if len(self.path) == 0:
            if self.location == self.target:
                self.finished = True
                if self.in_node:
                    sim.nodes.at[self.location, 'curr_capacity'] -= 1
                    self.in_node = False
            return self.x, self.y
        next_node_name = self.path[0]
        target_info = sim.nodes.loc[next_node_name]
        direct_x = target_info['x'] - self.x
        direct_y = target_info['y'] - self.y
        distance = np.sqrt(direct_x ** 2 + direct_y ** 2)
        if self.in_node and distance > threshold:  # 离开点,进入边
            sim.nodes.at[self.location, 'curr_capacity'] -= 1
            self.in_node = False
            if (self.location, next_node_name) in sim.edge_lookup:
                edge_idx = sim.edge_lookup[(self.location, next_node_name)]
                sim.edges.at[edge_idx, 'curr_capacity'] += 1
                self.current_edge_idx = edge_idx
        if distance < threshold:
            self.x = target_info['x']
            self.y = target_info['y']
            if not self.in_node:  # 离开边,进入点
                self.location = next_node_name
                sim.nodes.at[self.location, 'curr_capacity'] += 1
                self.in_node = True
                if self.current_edge_idx is not None:
                    sim.edges.at[self.current_edge_idx, 'curr_capacity'] -= 1
                    self.current_edge_idx = None
            self.path = plan_path(sim, self.location, self.target)[1:]
        elif distance > 0:
                normalized_dir_x = direct_x / distance
                normalized_dir_y = direct_y / distance
                self.x += normalized_dir_x * self.v
                self.y += normalized_dir_y * self.v
        return self.x, self.y

if __name__ == "__main__":
    sim.load_data(r'data\nodes.csv', r'data\edges.csv')
    app = QApplication(sys.argv)
    window = SubwayWindow(sim)
    window.show()
    sys.exit(app.exec())

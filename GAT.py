import pickle
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
plt.rcParams['font.sans-serif'] = ['SimHei']
import scipy.sparse as sp
import numpy as np
import torch
import os
import enum

class DatasetType(enum.Enum):
    Cora = 0

class GraphVisualizationTool(enum.Enum):
    NetworkX = 0,
    IGraph = 1

DATA_DIR_PATH = os.path.join(os.getcwd(), 'data')
CORA_PATH = os.path.join(DATA_DIR_PATH, 'cora')

CORA_TRAIN_RANGE = [0, 140]
CORA_VAL_RANGE = [140, 140+500]
CORA_TEST_RANGE = [1708, 1708+1000]
CORA_NUM_INPUT_FEATURES = 1433
CORA_NUM_CLASSES = 7

cora_label_to_color_map = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'orange',
    4: 'yellow',
    5: 'pink',
    6: 'gray'
}

def pickle_read(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def pickle_save(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# 我们稍后会传入训练配置字典
def load_graph_data(training_config, device):
    dataset_name = training_config['dataset_name'].lower()
    should_visualize = training_config['should_visualize']

    if dataset_name == DatasetType.Cora.name.lower():

        # shape = (N, FIN)，其中 N 是节点数，FIN 是输入特征数
        node_features_csr = pickle_read(os.path.join(CORA_PATH, 'node_features.csr'))
        # shape = (N, 1)
        node_labels_npy = pickle_read(os.path.join(CORA_PATH, 'node_labels.npy'))
        # shape = (N, 相邻节点数) <- 这是一个字典，不是矩阵！
        adjacency_list_dict = pickle_read(os.path.join(CORA_PATH, 'adjacency_list.dict'))

        # 标准化特征（有助于训练）
        node_features_csr = normalize_features_sparse(node_features_csr)
        num_of_nodes = len(node_labels_npy)

        # shape = (2, E)，其中 E 是边数，2 是源节点和目标节点。基本上边缘索引
        # 包含格式为 S->T 的元组，例如 0->3 表示具有 ID 0 的节点指向具有 ID 3 的节点。
        topology = build_edge_index(adjacency_list_dict, num_of_nodes, add_self_edges=True)

        # 注意：topology 只是命名图结构数据的花哨方式
        # （除边缘索引之外，它可以是邻接矩阵的形式）

        if should_visualize:  # 网络分析和图绘制
            plot_in_out_degree_distributions(topology, num_of_nodes, dataset_name)  # 我们将在第二部分定义这些
            visualize_graph(topology, node_labels_npy, dataset_name)

        # 转换为稠密 PyTorch 张量

        # 需要是 long int 类型，因为以后像 PyTorch 的 index_select 这样的函数期望它
        topology = torch.tensor(topology, dtype=torch.long, device=device)
        node_labels = torch.tensor(node_labels_npy, dtype=torch.long, device=device)  # 交叉熵期望一个 long int
        node_features = torch.tensor(node_features_csr.todense(), device=device)

        # 帮助我们提取属于训练/验证和测试拆分的节点的索引
        train_indices = torch.arange(CORA_TRAIN_RANGE[0], CORA_TRAIN_RANGE[1], dtype=torch.long, device=device)
        val_indices = torch.arange(CORA_VAL_RANGE[0], CORA_VAL_RANGE[1], dtype=torch.long, device=device)
        test_indices = torch.arange(CORA_TEST_RANGE[0], CORA_TEST_RANGE[1], dtype=torch.long, device=device)

        return node_features, node_labels, topology, train_indices, val_indices, test_indices
    else:
        raise Exception(f'{dataset_name} not yet supported.')

def normalize_features_sparse(node_features_sparse):
    assert sp.issparse(node_features_sparse), f'Expected a sparse matrix, got {node_features_sparse}.'

    # 而不是像 normalize_features_dense() 中那样进行除法，我们对特征的逆和进行乘法。
    # 现代硬件（GPU、TPU、ASIC）针对快速矩阵乘法进行了优化！ ^^ (* >> /)
    # 形状 = (N, FIN) -> (N, 1)，其中 N 表示节点数，FIN 表示输入特征数
    node_features_sum = np.array(node_features_sparse.sum(-1))  # 对每个节点特征向量求和特征

    # 创建一个逆矩阵（记住 * by 1/x 优于（更快）/ by x）
    # 形状 = (N, 1) -> (N)
    node_features_inv_sum = np.power(node_features_sum, -1).squeeze()

    # 再次某些和将为 0，因此 1/0 将为我们提供 inf，因此我们将它们替换为 1，它是 mul 的中性元素
    node_features_inv_sum[np.isinf(node_features_inv_sum)] = 1.

    # 创建一个对角矩阵，其对角线上的值来自 node_features_inv_sum
    diagonal_inv_features_sum_matrix = sp.diags(node_features_inv_sum)

    # 我们返回归一化的特征。
    return diagonal_inv_features_sum_matrix.dot(node_features_sparse)

def build_edge_index(adjacency_list_dict, num_of_nodes, add_self_edges=True):
    source_nodes_ids, target_nodes_ids = [], []
    seen_edges = set()

    for src_node, neighboring_nodes in adjacency_list_dict.items():
        for trg_node in neighboring_nodes:
            # if this edge hasn't been seen so far we add it to the edge index (coalescing - removing duplicates)
            if (src_node, trg_node) not in seen_edges:  # it'd be easy to explicitly remove self-edges (Cora has none..)
                source_nodes_ids.append(src_node)
                target_nodes_ids.append(trg_node)

                seen_edges.add((src_node, trg_node))

    if add_self_edges:
        source_nodes_ids.extend(np.arange(num_of_nodes))
        target_nodes_ids.extend(np.arange(num_of_nodes))

    # shape = (2, E), where E is the number of edges in the graph
    edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))

    return edge_index

# Let's just define dummy visualization functions for now - just to stop Python interpreter from complaining!

def plot_in_out_degree_distributions(edge_index, num_of_nodes, dataset_name):
    """
        注意：使用 igraph/networkx 等工具可以轻松进行各种强大的网络分析。
        我选择在此处显式计算仅节点度量统计，但如果需要，您可以深入研究并计算图直径、三角形数量以及许多其他网络分析领域的概念。

    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
        
    assert isinstance(edge_index, np.ndarray), f'Expected NumPy array got {type(edge_index)}.'

    # 存储每个节点的输入和输出度（对于无向图如 Cora，它们是相同的）
    in_degrees = np.zeros(num_of_nodes, dtype=int)
    out_degrees = np.zeros(num_of_nodes, dtype=int)

    # 边索引形状 = (2, E)，第一行包含源节点，第二行包含目标/汇节点
    # 术语说明：源节点指向目标/汇节点
    num_of_edges = edge_index.shape[1]
    for cnt in range(num_of_edges):
        source_node_id = edge_index[0, cnt]
        target_node_id = edge_index[1, cnt]

        out_degrees[source_node_id] += 1  # 源节点指向其他节点 -> 增加其出度
        in_degrees[target_node_id] += 1  # 类似地

    hist = np.zeros(np.max(out_degrees) + 1)
    for out_degree in out_degrees:
        hist[out_degree] += 1

    fig = plt.figure(figsize=(12,8), dpi=100)  # 否则在 Jupyter Notebook 中图表会很小
    fig.subplots_adjust(hspace=0.6)

    plt.subplot(311)
    plt.plot(in_degrees, color='red')
    plt.xlabel('node id'); plt.ylabel('in-degree count'); plt.title('不同节点 id 的输入度')

    plt.subplot(312)
    plt.plot(out_degrees, color='green')
    plt.xlabel('node id'); plt.ylabel('out-degree count'); plt.title('不同节点 id 的输出度')

    plt.subplot(313)
    plt.plot(hist, color='blue')
    plt.xlabel('node degree')
    plt.ylabel('给定出度的节点数量') 
    plt.title(f'{dataset_name} 数据集的节点出度分布')
    plt.xticks(np.arange(0, len(hist), 5.0))

    plt.grid(True)
    plt.show()



def visualize_graph():
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU

config = {
    'dataset_name': DatasetType.Cora.name,
    'should_visualize': False
}

node_features, node_labels, edge_index, train_indices, val_indices, test_indices = load_graph_data(config, device)

print(node_features.shape, node_features.dtype)
print(node_labels.shape, node_labels.dtype)
print(edge_index.shape, edge_index.dtype)
print(train_indices.shape, train_indices.dtype)
print(val_indices.shape, val_indices.dtype)
print(test_indices.shape, test_indices.dtype)

num_of_nodes = len(node_labels)
#plot_in_out_degree_distributions(edge_index, num_of_nodes, config['dataset_name'])

"""
请参阅此博客以了解可用的图形可视化工具：
  https://towardsdatascience.com/large-graph-visualization-tools-and-approaches-2b8758a1cd59

基本上，取决于您的图形大小，可能会有一些比 igraph 更好的绘图工具。

注意：不幸的是，我不得不将此函数扁平化，因为 igraph 在 Jupyter Notebook 中遇到了一些问题，
我们只会在这里调用它，所以没关系！

"""

dataset_name = config['dataset_name']
visualization_tool=GraphVisualizationTool.IGraph

if isinstance(edge_index, torch.Tensor):
    edge_index_np = edge_index.cpu().numpy()

if isinstance(node_labels, torch.Tensor):
    node_labels_np = node_labels.cpu().numpy()

num_of_nodes = len(node_labels_np)
edge_index_tuples = list(zip(edge_index_np[0, :], edge_index_np[1, :]))  # igraph 要求这种格式

# 构建 igraph 图
ig_graph = ig.Graph()
ig_graph.add_vertices(num_of_nodes)
ig_graph.add_edges(edge_index_tuples)

# 准备可视化设置字典
visual_style = {
    "bbox": (700, 700),
    "margin": 5,
}

# 我选择边的厚度与通过我们图中某个边的最短路径（测地线）的数量成比例（edge_betweenness 函数，一个简单的 ad hoc 启发式）

# line1：我使用日志，否则一些边会太厚，而其他边根本不明显
# edge_betweenness 返回 < 1 对于某些边，这就是为什么我使用 clip 作为 log 对于那些边来说是负的
# line2：归一化，使最厚的边为 1，否则边在图表上看起来太厚
# line3：这里的想法是让最强的边缘保持比其他边缘更强，6 刚刚好，不要纠结于此

edge_weights_raw = np.clip(np.log(np.asarray(ig_graph.edge_betweenness())+1e-16), a_min=0, a_max=None)
edge_weights_raw_normalized = edge_weights_raw / np.max(edge_weights_raw)
edge_weights = [w**6 for w in edge_weights_raw_normalized]
visual_style["edge_width"] = edge_weights

# 顶点大小的简单启发式。大小 ~ (度/4)（我尝试了 log 和 sqrt 也取得了很好的效果）
visual_style["vertex_size"] = [deg / 4 for deg in ig_graph.degree()]

# Cora 特有的部分，因为 Cora 有 7 个标签
if dataset_name.lower() == DatasetType.Cora.name.lower():
    visual_style["vertex_color"] = [cora_label_to_color_map[label] for label in node_labels_np]
else:
    print('请随意为您的特定数据集添加自定义配色方案。使用 igraph 默认配色。')

# 设置布局 - 图表在 2D 图表上呈现的方式。图形绘制本身是一个子领域！
# 我使用“Kamada Kawai”力导向方法，这组方法基于物理系统模拟。
# （layout_drl 也为 Cora 提供了不错的结果）
visual_style["layout"] = ig_graph.layout("kk")  # Kamada Kawai
# visual_style["layout"] = ig_graph.layout("drl")  # Distributed Recursive Layout

print('正在绘制结果...（可能需要几秒钟）。')
ig.plot(ig_graph, **visual_style)

# 这个网站有一些很棒的可视化效果，请查看：
# http://networkrepository.com/graphvis.php?d=./data/gsm50/labeled/cora.edges


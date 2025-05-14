import torch
import torch.nn as nn
from torch.optim import Adam


class GAT(torch.nn.Module):
    """
    最有趣和最难的实现是实现＃3。
    Imp1和imp2在细节上有所不同，但基本上是相同的东西。

    因此，我将在本笔记本中专注于imp＃3。

    """

    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6, log_attention_weights=False):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'输入有效的架构参数。'

        num_heads_per_layer = [1] + num_heads_per_layer  # 技巧-这样我可以很好地创建下面的GAT层

        gat_layers = []  # 收集GAT层
        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # 连接的结果
                num_out_features=num_features_per_layer[i+1],
                num_of_heads=num_heads_per_layer[i+1],
                concat=True if i < num_of_layers - 1 else False,  # 最后一个GAT层执行平均值，其他层执行连接
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # 最后一层只输出原始分数
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    # 数据只是一个（in_nodes_features，edge_index）元组，我必须这样做是因为nn.Sequential：
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
    def forward(self, data):
        return self.gat_net(data)

class GATLayer(torch.nn.Module):
    """
    实现 #3 受到 PyTorch Geometric 启发：https://github.com/rusty1s/pytorch_geometric

    但是，这里的实现应该更容易理解！（并且性能相似）

    """
    
    # 我们会在许多函数中使用这些常量，所以在这里提取为成员字段
    src_nodes_dim = 0  # 边索引中源节点的位置
    trg_nodes_dim = 1  # 边索引中目标节点的位置

    # 在归纳设置中，这些可能会改变 - 暂时保留这样的设置（未来可能不适用）
    nodes_dim = 0      # 节点维度（轴在张量中可能是一个更熟悉的术语，节点维度是"N"的位置）
    head_dim = 1       # 注意力头维度

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # 是否应该连接还是平均注意力头
        self.add_skip_connection = add_skip_connection

        #
        # 可训练权重：线性投影矩阵（在论文中表示为"W"）、注意力目标/源（在论文中表示为"a"）和偏差（在论文中未提到，但在官方GAT存储库中存在）
        #

        # 可以将这个矩阵视为 num_of_heads 个独立的 W 矩阵
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # 在我们连接目标节点（节点 i）和源节点（节点 j）之后，我们应用“加法”评分函数
        # 它给我们未标准化的分数 "e"。在这里，我们分割 "a" 向量 - 但语义保持不变。
        # 基本上，与执行 [x, y]（连接，x/y 是节点特征向量）和与 "a" 的点积不同，
        # 我们分别对 x 和 "a_left" 进行点积，对 y 和 "a_right" 进行点积，然后将它们相加
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        # 在 GAT 中偏置绝对不是关键的 - 随时实验（我在这个问题上向主要作者 Petar 询问过）
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # 可训练权重结束
        #

        self.leakyReLU = nn.LeakyReLU(0.2)  # 使用 0.2，就像在论文中一样，不需要公开每个设置
        self.activation = activation
        # 可能不是最好的设计，但我在 3 个位置使用相同的模块，用于特征投影之前/之后和注意力系数。
        # 就功能而言，它与使用独立模块是相同的。
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # 是否应记录注意力权重
        self.attention_weights = None  # 用于后续可视化目的，我在这里缓存权重

        self.init_params()
        
    def forward(self, data):
        #
        # 步骤 1：线性投影 + 正则化
        #

        in_nodes_features, edge_index = data  # 解包数据
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'期望形状为 (2,E) 的边索引，得到了 {edge_index.shape}'

        # 形状 = (N, FIN)，其中 N 是图中的节点数，FIN 是每个节点的输入特征数
        # 我们对所有输入节点特征应用 dropout（正如论文中所提到的）
        # 注意：对于 Cora，特征已经非常稀疏，所以实际上可能帮助不大
        in_nodes_features = self.dropout(in_nodes_features)

        # 形状 = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT)，其中 NH 是注意力头的数量，FOUT 是输出特征的数量
        # 我们将输入节点特征投影到 NH 个独立的输出特征中（每个注意力头一个）
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # 在官方 GAT 实现中，他们在这里也使用了 dropout

        #
        # 步骤 2：边注意力计算
        #

        # 应用评分函数（* 表示按元素（也称为Hadamard）乘法）
        # 形状 = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH)，因为 sum 压缩了最后一个维度
        # 优化注：在我的实验中，torch.sum() 的性能与 .sum() 一样好
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # 我们只需根据边索引复制（提升）源/目标节点的分数。我们不需要准备所有可能的分数组合，
        # 我们只需要准备那些将实际使用的分数组合，这由边索引定义
        # 分数形状 = (E, NH)，nodes_features_proj_lifted 形状 = (E, NH, FOUT)，E 是图中的边数
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # 形状 = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        # 对邻居聚合添加随机性
        attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # 步骤 3：邻居聚合
        #

        # 逐元素（也称为Hadamard）乘法。运算符 * 执行与 torch.mul 相同的操作
        # 形状 = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT)，1 被广播到 FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # 这一部分对每个目标节点累积加权和投影的邻居特征向量
        # 形状 = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        #
        # 步骤 4：残差/跳跃连接、连接和偏差
        #

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index)

    #
    # 辅助函数（没有注释几乎没有代码，所以不要害怕！）
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        正如函数名所示，它对邻居执行 softmax。例如：假设图中有 5 个节点。其中的两个节点 1、2 与节点 3 相连。
        如果我们要计算节点 3 的表示，我们应该考虑节点 1、2 和节点 3 本身的特征向量。由于我们对边 1-3、2-3 和 3-3 的分数
        进行了评估，这个函数将计算类似这样的注意力分数：1-3 / (1-3 + 2-3 + 3-3)（其中 1-3 是过载的符号，它表示边 1-3 及其（exp）分数），
        类似地对于 2-3 和 3-3，即对于这个邻居，我们不关心包含节点 4 和 5 的其他边分数。

        注意：
        从 logits 中减去最大值不会改变最终结果，但它提高了数值稳定性，并且在几乎每个深度学习框架中，这是一个相当常见的“技巧”。
        有关更多详细信息，请查看此链接：

        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        """
        # 计算分子。使 logits <= 0，以便 e^logit <= 1（这将提高数值稳定性）
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # 计算分母。形状 = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 在理论上不是必需的，但它仅出于数值稳定性考虑存在（避免除以 0）- 由于计算机将非常小的数字四舍五入到 0，这是可能的
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # 形状必须与 exp_scores_per_edge 相同（由 scatter_add_ 要求），即从 E 变为 (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # 形状为 (N, NH)，其中 N 是节点数量，NH 是注意力头的数量
        size = list(exp_scores_per_edge.shape)  # 转换为列表，否则无法进行赋值
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # 位置 i 包含所有指向节点 i 的节点的 exp 分数之和（由目标索引指定）
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # 再次扩展，以便将其用作 softmax 分母。例如，节点 i 的总和将复制到源节点指向 i 的所有位置（由目标索引指定）
        # 形状为 (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # 转换为列表，否则无法进行赋值
        size[self.nodes_dim] = num_of_nodes  # 形状为 (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # 形状为 (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        # 聚合步骤 - 我们累积所有注意力头的投影加权节点特征
        # 形状为 (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        抬升（Lift）即根据边索引复制特定向量。
        张量的维度之一从 N 变为 E（这就是“抬升”一词的来源）。

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # 使用 index_select 比在 PyTorch 中使用 "normal" 索引（scores_source[src_nodes_index]）更快！
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # 附加单例维度，直到 this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # 明确扩展以使形状相同
        return this.expand_as(other)

    def init_params(self):
        """
        我们使用 Glorot（也称为 Xavier 均匀）初始化的原因是因为它是 TF 的默认初始化方式：
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        原始库在 TensorFlow（TF）中开发，并且他们使用了默认初始化。
        随时进行实验 - 根据问题可能有更好的初始化方法。

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # 可能记录以供稍后在 playground.py 中可视化
            self.attention_weights = attention_coefficients

        if self.add_skip_connection:  # 添加跳跃或残差连接
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # 如果 FIN == FOUT
                # unsqueeze 实现以下效果：(N, FIN) -> (N, 1, FIN)，输出特征为 (N, NH, FOUT) 所以 1 被广播到 NH
                # 因此，基本上我们将输入向量 NH 次复制并添加到处理过的向量中
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT，因此我们需要将输入特征向量投影到可以添加到输出特征向量的维度。
                # skip_proj 添加了大量额外的容量，这可能导致过拟合。
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            # 形状为 (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # 形状为 (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)

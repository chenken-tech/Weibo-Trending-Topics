import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv


class GCNModel(nn.Module):
    """
    基于 GCN 的异构图回归模型。
    将异构图转为同构图，对节点特征进行投影后应用多层 GCN 卷积，
    最后提取 topic 节点特征进行回归预测。
    """
    def __init__(self, graph, input_dim_dict, hidden_dim=32, n_layers=2, dropout=0.4):
        super().__init__()
        # 节点类型投影：将不同输入维度的特征映射到 hidden_dim
        self.proj = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim)
            for ntype, in_dim in input_dim_dict.items()
        })

        # 将异构图转为同构图并缓存
        self.homo_g, self._node_range = self._build_cache(graph)

        # 堆叠 GCN 层
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)

        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def _build_cache(self, hetero_graph):
        """将异构图转为同构图，并记录 topic 节点在全局节点中的索引范围"""
        node_offsets = {}
        total_nodes = 0
        node_range = {}
        for ntype in hetero_graph.ntypes:
            node_offsets[ntype] = total_nodes
            num = hetero_graph.number_of_nodes(ntype)
            node_range[ntype] = (total_nodes, total_nodes + num)
            total_nodes += num

        src_list, dst_list = [], []
        for stype, etype, dtype in hetero_graph.canonical_etypes:
            subg = hetero_graph[stype, etype, dtype]
            if subg.number_of_edges() == 0:
                continue
            u, v = subg.edges()
            src_list.append(u + node_offsets[stype])
            dst_list.append(v + node_offsets[dtype])
        src = torch.cat(src_list)
        dst = torch.cat(dst_list)
        homo_g = dgl.graph((src, dst), num_nodes=total_nodes)
        return homo_g, node_range

    def forward(self, g, device):
        g = g.to(device)
        homo_g = self.homo_g.to(device)

        # 1. 投影所有节点特征
        all_feats = []
        for ntype in g.ntypes:
            feat = g.nodes[ntype].data['feat'].to(device)
            all_feats.append(self.proj[ntype](feat))
        x = torch.cat(all_feats, dim=0)  # (total_nodes, hidden_dim)

        # 2. 多层 GCN
        for layer in self.layers:
            x = layer(homo_g, x)
            x = F.relu(x)
            x = self.dropout(x)

        # 3. 提取 topic 节点特征
        start, end = self._node_range['topic']
        topic_emb = x[start:end]

        # 4. 回归预测
        pred = self.regressor(topic_emb)
        return pred, {'topic': topic_emb}
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import dgl
# import dgl.function as fn
#
#
# class RGCNLayer(nn.Module):
#     """
#     单层 RGCN：对每种关系学习单独的线性变换，然后按关系聚合邻居消息。
#     输出：对每种目标节点类型，收集所有入边关系的变换后消息，求和（或平均），再加自环。
#     """
#     def __init__(self, node_in_dims, hidden_dim, graph, activation=F.relu, dropout=0.0, self_loop=True):
#         super().__init__()
#         self.node_in_dims = node_in_dims
#         self.hidden_dim = hidden_dim
#         self.num_relations = len(graph.canonical_etypes)
#         self.activation = activation
#         self.dropout = nn.Dropout(dropout)
#         self.self_loop = self_loop
#
#         # 每个节点类型都有自己的线性变换（将输入维度映射到 hidden_dim）
#         self.node_proj = nn.ModuleDict({
#             ntype: nn.Linear(in_dim, hidden_dim, bias=False)
#             for ntype, in_dim in node_in_dims.items()
#         })
#         # 每个关系类型也有独立的线性变换（映射源节点特征到 hidden_dim）
#         self.rel_proj = nn.ModuleList([
#             nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(self.num_relations)
#         ])
#         if self.self_loop:
#             self.loop_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
#
#         # 建立边类型三元组到索引的映射
#         self.rel_to_idx = {triplet: idx for idx, triplet in enumerate(graph.canonical_etypes)}
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         gain = nn.init.calculate_gain('relu')
#         for proj in self.node_proj.values():
#             nn.init.xavier_normal_(proj.weight, gain=gain)
#         for proj in self.rel_proj:
#             nn.init.xavier_normal_(proj.weight, gain=gain)
#         if self.self_loop:
#             nn.init.xavier_normal_(self.loop_proj.weight, gain=gain)
#
#     def forward(self, graph, node_feats):
#         # 第一步：对所有节点进行初步投影
#         h = {ntype: self.node_proj[ntype](node_feats[ntype]) for ntype in node_feats}
#         # 存储每种节点类型的最终聚合结果
#         out = {ntype: torch.zeros(graph.number_of_nodes(ntype), self.hidden_dim, device=graph.device)
#                for ntype in graph.ntypes}
#
#         for stype, etype, dtype in graph.canonical_etypes:
#             triplet = (stype, etype, dtype)
#             rel_idx = self.rel_to_idx[triplet]
#             subg = graph[stype, etype, dtype]
#             if subg.number_of_edges() == 0:
#                 continue
#             src_feat = h[stype]          # (N_src, hidden_dim)
#             # 关系变换
#             msg = self.rel_proj[rel_idx](src_feat)   # (N_src, hidden_dim)
#             # 按目标节点聚合（求和）
#             subg = subg.local_var()
#             subg.srcdata['msg'] = msg
#             subg.update_all(fn.copy_u('msg', 'm'), fn.sum('m', 'agg'))
#             agg = subg.dstdata['agg']    # (N_dst, hidden_dim)
#             out[dtype] = out[dtype] + agg
#
#         # 自环（如果启用）
#         if self.self_loop:
#             for ntype in graph.ntypes:
#                 out[ntype] = out[ntype] + self.loop_proj(h[ntype])
#
#         # 可选的激活和 dropout
#         for ntype in out:
#             out[ntype] = self.dropout(self.activation(out[ntype]))
#         return out
#
#
# class RGCNEncoder(nn.Module):
#     """堆叠多层 RGCN 层"""
#     def __init__(self, graph, node_in_dims, hidden_dim, num_layers, dropout=0.0, self_loop=True):
#         super().__init__()
#         self.num_layers = num_layers
#         self.layers = nn.ModuleList()
#
#         for i in range(num_layers):
#             # 第一层使用原始的 node_in_dims，后续层使用 hidden_dim
#             in_dims = node_in_dims if i == 0 else {ntype: hidden_dim for ntype in graph.ntypes}
#             layer = RGCNLayer(in_dims, hidden_dim, graph, activation=F.relu, dropout=dropout, self_loop=self_loop)
#             self.layers.append(layer)
#
#     def forward(self, blocks, node_feats):
#         h = node_feats
#         for i, layer in enumerate(self.layers):
#             g = blocks[i] if i < len(blocks) else blocks[0]
#             h = layer(g, h)
#         return h
#
#
# class GlobalTopicRegressor_RGCN(nn.Module):
#     def __init__(self, graph, input_dim_dict, hidden_dim=64, n_layers=2, dropout=0.5, self_loop=True):
#         super().__init__()
#         self.encoder = RGCNEncoder(graph, input_dim_dict, hidden_dim, n_layers, dropout, self_loop)
#         self.regressor = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 1)
#         )
#
#     def encode(self, g, device):
#         node_features = {ntype: g.nodes[ntype].data["feat"].to(device) for ntype in g.ntypes}
#         blocks = [g.to(device) for _ in range(self.encoder.num_layers)]
#         return self.encoder(blocks, node_features)
#
#     def forward(self, g, device):
#         h = self.encode(g, device)
#         pred = self.regressor(h["topic"])
#         return pred, h



import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn


class RGCNLayer(nn.Module):
    """
    改进版 RGCN 层：
    - 对每种关系进行线性变换，按目标节点入度平均聚合
    - 添加层归一化 (LayerNorm)
    - 自环单独变换后与邻居消息相加
    """
    def __init__(self, in_dims, out_dim, graph, activation=F.relu, dropout=0.0, self_loop=True, norm=True):
        super().__init__()
        self.out_dim = out_dim
        self.num_rels = len(graph.canonical_etypes)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.self_loop = self_loop
        self.norm = norm

        # 节点类型输入投影（输入维度 -> out_dim）
        self.node_proj = nn.ModuleDict({
            ntype: nn.Linear(in_dims[ntype], out_dim, bias=True) for ntype in in_dims
        })
        # 关系投影矩阵（每种关系一个线性层）
        self.rel_proj = nn.ModuleList([
            nn.Linear(out_dim, out_dim, bias=True) for _ in range(self.num_rels)
        ])
        if self_loop:
            self.loop_proj = nn.Linear(out_dim, out_dim, bias=True)
        if norm:
            self.layer_norm = nn.ModuleDict({
                ntype: nn.LayerNorm(out_dim) for ntype in in_dims
            })

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.node_proj.values():
            nn.init.xavier_uniform_(p.weight)
            if p.bias is not None:
                nn.init.zeros_(p.bias)
        for p in self.rel_proj:
            nn.init.xavier_uniform_(p.weight)
            if p.bias is not None:
                nn.init.zeros_(p.bias)
        if self.self_loop:
            nn.init.xavier_uniform_(self.loop_proj.weight)
            if self.loop_proj.bias is not None:
                nn.init.zeros_(self.loop_proj.bias)

    def forward(self, graph, feats):
        # 1. 对每个节点类型做线性投影
        h = {nt: self.node_proj[nt](feats[nt]) for nt in feats}
        # 2. 初始化输出为 0
        out = {nt: torch.zeros(graph.num_nodes(nt), self.out_dim, device=graph.device)
               for nt in graph.ntypes}

        # 建立关系三元组到索引的映射
        rel_index = {triplet: i for i, triplet in enumerate(graph.canonical_etypes)}

        # 3. 按关系类型传播消息
        for st, et, dt in graph.canonical_etypes:
            idx = rel_index[(st, et, dt)]
            subg = graph[st, et, dt]
            if subg.num_edges() == 0:
                continue
            # 对源节点特征做关系变换
            msg = self.rel_proj[idx](h[st])
            subg = subg.local_var()
            subg.srcdata['msg'] = msg
            # 归一化因子：1 / 目标节点入度（避免度数大的节点主导）
            in_deg = subg.in_degrees().float().clamp(min=1).to(msg.device)
            subg.edata['norm'] = 1.0 / in_deg[subg.edges()[1]]
            subg.update_all(fn.u_mul_e('msg', 'norm', 'm'), fn.sum('m', 'agg'))
            out[dt] = out[dt] + subg.dstdata['agg']

        # 4. 自环
        if self.self_loop:
            for nt in graph.ntypes:
                out[nt] = out[nt] + self.loop_proj(h[nt])

        # 5. 激活、dropout、可选的层归一化
        for nt in out:
            out[nt] = self.dropout(self.activation(out[nt]))
            if self.norm:
                out[nt] = self.layer_norm[nt](out[nt])
        return out


class RGCNEncoder(nn.Module):
    """堆叠多层 RGCN"""
    def __init__(self, graph, in_dims, hidden_dim, num_layers, dropout=0.0, self_loop=True):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        current_dims = in_dims
        for _ in range(num_layers):
            layer = RGCNLayer(current_dims, hidden_dim, graph,
                              activation=F.relu, dropout=dropout,
                              self_loop=self_loop, norm=True)
            self.layers.append(layer)
            # 下一层的输入维度全部变为 hidden_dim（所有节点类型输出维数相同）
            current_dims = {nt: hidden_dim for nt in graph.ntypes}

    def forward(self, blocks, feats):
        h = feats
        for i, layer in enumerate(self.layers):
            g = blocks[i] if i < len(blocks) else blocks[0]
            h = layer(g, h)
        return h


class GlobalTopicRegressor_RGCN(nn.Module):
    def __init__(self, graph, input_dim_dict, hidden_dim=64, n_layers=2, dropout=0.5, self_loop=True):
        super().__init__()
        self.encoder = RGCNEncoder(graph, input_dim_dict, hidden_dim, n_layers, dropout, self_loop)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def encode(self, g, device):
        node_features = {nt: g.nodes[nt].data["feat"].to(device) for nt in g.ntypes}
        blocks = [g.to(device) for _ in range(self.encoder.num_layers)]
        return self.encoder(blocks, node_features)

    def forward(self, g, device):
        h = self.encode(g, device)
        pred = self.regressor(h["topic"])
        return pred, h
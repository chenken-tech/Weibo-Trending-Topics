import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.ops import edge_softmax
import dgl.function as fn


class RelationGAT(nn.Module):
    """单个关系上的多头图注意力层（节点级注意力）。输出拼接的多头特征。"""
    def __init__(self, in_src_dim, in_dst_dim, out_dim, num_heads, dropout=0.0, negative_slope=0.2):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.W_src = nn.Linear(in_src_dim, num_heads * out_dim, bias=False)
        self.W_dst = nn.Linear(in_dst_dim, num_heads * out_dim, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, num_heads, 2 * out_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W_src.weight, gain=gain)
        nn.init.xavier_normal_(self.W_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.att, gain=gain)

    def forward(self, graph, feat_src, feat_dst):
        with graph.local_scope():
            h_src = self.W_src(feat_src).view(-1, self.num_heads, self.out_dim)
            h_dst = self.W_dst(feat_dst).view(-1, self.num_heads, self.out_dim)
            graph.srcdata['h'] = h_src
            graph.dstdata['h_dst'] = h_dst

            def concat_udf(edges):
                return {'cat_h': torch.cat([edges.src['h'], edges.dst['h_dst']], dim=-1)}

            graph.apply_edges(concat_udf)
            e = (graph.edata['cat_h'] * self.att).sum(dim=-1, keepdim=True)
            e = self.leaky_relu(e)
            graph.edata['a'] = edge_softmax(graph, e)
            graph.update_all(fn.u_mul_e('h', 'a', 'msg'), fn.sum('msg', 'h'))
            ret = graph.dstdata['h']           # (N_dst, heads, out_dim)
            ret = ret.view(-1, self.num_heads * self.out_dim)   # 拼接多头
            return ret


class SemanticAttention(nn.Module):
    """语义级注意力：对多个关系表示进行加权融合"""
    def __init__(self, in_dim, num_relations, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.Parameter(torch.Tensor(num_heads, in_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.attention)

    def forward(self, inputs):
        # inputs: list of (N, D) tensors
        inputs = torch.stack(inputs, dim=0)         # (R, N, D)
        if self.num_heads > 1:
            # 简化：只支持单头语义注意力
            pass
        # 计算注意力分数 (R, N)
        att_score = torch.einsum('hd, rnd -> rn', self.attention, inputs)
        att_score = att_score.permute(1, 0)         # (N, R)
        att_score = F.softmax(att_score, dim=-1)    # (N, R)
        # 加权融合 (N, D)
        out = torch.einsum('nr, rnd -> nd', att_score, inputs)
        return out


class HANLayer(nn.Module):
    """单层HAN：对每种目标节点类型，聚合所有入边关系并用语义注意力融合"""
    def __init__(self, graph, node_in_dims, hidden_dim, num_heads, dropout=0.0, residual=True):
        super().__init__()
        self.node_types = graph.ntypes
        self.etypes = graph.canonical_etypes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.residual = residual

        # 按目标节点类型收集关系
        dst2relations = {}
        for st, et, dt in self.etypes:
            dst2relations.setdefault(dt, []).append((st, et, dt))

        self.relation_gats = nn.ModuleDict()
        self.semantic_attns = nn.ModuleDict()
        # 存储每个目标节点类型对应的关系三元组列表
        self.rel_triplets = {}   # dst_ntype -> list of (st, et, dt)

        for dst_ntype in self.node_types:
            rel_list = dst2relations.get(dst_ntype, [])
            if not rel_list:
                continue
            gats = nn.ModuleDict()
            triplets = []
            for st, et, dt in rel_list:
                in_src = node_in_dims[st]
                in_dst = node_in_dims[dst_ntype]
                # 使用竖线分隔避免下划线混淆
                rel_id = f"{st}|{et}|{dt}"
                gats[rel_id] = RelationGAT(in_src, in_dst, hidden_dim, num_heads, dropout)
                triplets.append((st, et, dt))
            self.relation_gats[dst_ntype] = gats
            self.rel_triplets[dst_ntype] = triplets
            self.semantic_attns[dst_ntype] = SemanticAttention(hidden_dim * num_heads, len(rel_list))

        if self.residual:
            self.res_fc = nn.ModuleDict()
            for ntype in self.node_types:
                if ntype in node_in_dims:
                    self.res_fc[ntype] = nn.Linear(node_in_dims[ntype], hidden_dim * num_heads, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.elu

    def forward(self, graph, node_feats):
        new_feats = {}
        for dst_ntype, gats in self.relation_gats.items():
            rel_outputs = []
            triplets = self.rel_triplets[dst_ntype]
            # 同时遍历模块和三元组
            for (st, et, dt), gat in zip(triplets, gats.values()):
                subg = graph[st, et, dt]
                if subg.number_of_edges() == 0:
                    continue
                src_feat = node_feats[st]
                dst_feat = node_feats[dst_ntype]
                out = gat(subg, src_feat, dst_feat)
                rel_outputs.append(out)
            if rel_outputs:
                fused = self.semantic_attns[dst_ntype](rel_outputs)
                if self.residual and dst_ntype in self.res_fc:
                    fused = fused + self.res_fc[dst_ntype](node_feats[dst_ntype])
                fused = self.dropout(self.activation(fused))
                new_feats[dst_ntype] = fused
            else:
                new_feats[dst_ntype] = node_feats[dst_ntype]

        # 没有入边的节点类型直接传递特征
        for ntype in self.node_types:
            if ntype not in new_feats:
                new_feats[ntype] = node_feats[ntype]
        return new_feats


class HANEncoder(nn.Module):
    """堆叠多层 HANLayer"""
    def __init__(self, graph, node_in_dims, hidden_dim, num_layers, num_heads, dropout=0.0, residual=True):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        current_dims = node_in_dims
        for _ in range(num_layers):
            layer = HANLayer(graph, current_dims, hidden_dim, num_heads, dropout, residual)
            self.layers.append(layer)
            # 更新下一层的输入维度
            new_dims = {ntype: hidden_dim * num_heads for ntype in node_in_dims if ntype in graph.ntypes}
            for ntype in graph.ntypes:
                if ntype not in new_dims:
                    new_dims[ntype] = current_dims.get(ntype, 0)
            current_dims = new_dims
        self.out_dims = current_dims

    def forward(self, blocks, node_feats):
        h = node_feats
        for i, layer in enumerate(self.layers):
            g = blocks[i] if i < len(blocks) else blocks[0]
            h = layer(g, h)
        return h


# class GlobalTopicRegressor_HAN(nn.Module):
#     def __init__(self, graph, input_dim_dict, hidden_dim=64, n_layers=2, n_heads=2, dropout=0.5):
#         super().__init__()
#         self.encoder = HANEncoder(graph, input_dim_dict, hidden_dim, n_layers, n_heads, dropout, residual=True)
#         out_dim = hidden_dim * n_heads
#         self.regressor = nn.Sequential(
#             nn.Linear(out_dim, out_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(out_dim, 1)
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

class GlobalTopicRegressor_HAN(nn.Module):
    def __init__(self, graph, input_dim_dict, hidden_dim=64, n_layers=2, n_heads=2, dropout=0.5):
        super().__init__()
        self.encoder = HANEncoder(graph, input_dim_dict, hidden_dim, n_layers, n_heads, dropout, residual=True)
        out_dim = hidden_dim * n_heads
        self.regressor = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1)
        )
        # 可选：输出缩放参数
        self.out_scale = nn.Parameter(torch.ones(1))
        self.out_bias = nn.Parameter(torch.zeros(1))

    def encode(self, g, device):
        node_features = {ntype: g.nodes[ntype].data["feat"].to(device) for ntype in g.ntypes}
        blocks = [g.to(device) for _ in range(self.encoder.num_layers)]
        return self.encoder(blocks, node_features)

    def forward(self, g, device):
        h = self.encode(g, device)
        pred = self.regressor(h["topic"])
        pred = pred * self.out_scale + self.out_bias   # 缩放偏移（可选）
        return pred, h
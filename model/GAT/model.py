import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax


class GATLayer(nn.Module):
    """支持节点类型感知的多头图注意力层"""
    def __init__(self, in_dim, out_dim, num_ntypes, n_heads, dropout=0.2, negative_slope=0.2, use_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_ntypes = num_ntypes
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.use_norm = use_norm
        self.negative_slope = negative_slope

        self.fc = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_ntypes)])
        self.att_l = nn.ParameterList([nn.Parameter(torch.Tensor(n_heads, self.d_k)) for _ in range(num_ntypes)])
        self.att_r = nn.ParameterList([nn.Parameter(torch.Tensor(n_heads, self.d_k)) for _ in range(num_ntypes)])
        self.norms = nn.ModuleList([nn.LayerNorm(out_dim) if use_norm else nn.Identity() for _ in range(num_ntypes)])
        self.skip = nn.Parameter(torch.ones(num_ntypes))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_ntypes):
            nn.init.xavier_uniform_(self.fc[i].weight)
            nn.init.xavier_uniform_(self.att_l[i])
            nn.init.xavier_uniform_(self.att_r[i])
            if isinstance(self.norms[i], nn.LayerNorm):
                nn.init.constant_(self.norms[i].weight, 1)
                nn.init.constant_(self.norms[i].bias, 0)

    def forward(self, g, x, ntype_ids):
        with g.local_scope():
            # 1) 投影
            proj_x = torch.zeros(x.size(0), self.out_dim, device=x.device)
            for t in range(self.num_ntypes):
                mask = (ntype_ids == t)
                if mask.any():
                    proj_x[mask] = self.fc[t](x[mask])
            h = proj_x.view(-1, self.n_heads, self.d_k)

            # 2) 注意力系数
            alpha_src = torch.zeros(x.size(0), self.n_heads, device=x.device)
            alpha_dst = torch.zeros(x.size(0), self.n_heads, device=x.device)
            for t in range(self.num_ntypes):
                mask = (ntype_ids == t)
                if mask.any():
                    h_t = h[mask]
                    alpha_src[mask] = (h_t * self.att_l[t]).sum(dim=-1)
                    alpha_dst[mask] = (h_t * self.att_r[t]).sum(dim=-1)
            g.ndata['h'] = h
            g.ndata['alpha_src'] = alpha_src
            g.ndata['alpha_dst'] = alpha_dst

            g.apply_edges(fn.u_add_v('alpha_src', 'alpha_dst', 'e'))
            e = F.leaky_relu(g.edata['e'], negative_slope=self.negative_slope)
            a = edge_softmax(g, e, norm_by='dst')
            g.edata['a'] = a.unsqueeze(-1)

            g.update_all(fn.u_mul_e('h', 'a', 'msg'), fn.sum('msg', 'agg'))
            agg = g.ndata['agg'].view(-1, self.out_dim)

            # 3) 残差 + 归一化
            for t in range(self.num_ntypes):
                mask = (ntype_ids == t)
                if mask.any():
                    alpha = torch.sigmoid(self.skip[t])
                    agg[mask] = self.dropout(agg[mask]) * alpha + proj_x[mask] * (1 - alpha)
                    agg[mask] = self.norms[t](agg[mask])
            return agg


class GATModel(nn.Module):
    def __init__(self, graph, input_dim_dict, hidden_dim=32, n_layers=2, n_heads=2, dropout=0.4):
        super().__init__()
        self.num_ntypes = len(graph.ntypes)
        self.proj = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim) for ntype, in_dim in input_dim_dict.items()
        })
        self.homo_g, self.ntype_ids, self._node_range = self._build_cache(graph)

        self.layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim, self.num_ntypes, n_heads, dropout, use_norm=True)
            for _ in range(n_layers)
        ])
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def _build_cache(self, hetero_graph):
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

        ntype_ids_list = []
        for ntype in hetero_graph.ntypes:
            num = hetero_graph.number_of_nodes(ntype)
            ntype_id = hetero_graph.ntypes.index(ntype)
            ntype_ids_list.append(torch.full((num,), ntype_id, dtype=torch.long))
        ntype_ids = torch.cat(ntype_ids_list, dim=0)
        return homo_g, ntype_ids, node_range

    def forward(self, g, device):
        g = g.to(device)
        homo_g = self.homo_g.to(device)
        ntype_ids = self.ntype_ids.to(device)

        all_feats = []
        for ntype in g.ntypes:
            feat = g.nodes[ntype].data['feat'].to(device)
            all_feats.append(self.proj[ntype](feat))
        x = torch.cat(all_feats, dim=0)

        for layer in self.layers:
            x = layer(homo_g, x, ntype_ids)

        start, end = self._node_range['topic']
        topic_emb = x[start:end]
        pred = self.regressor(topic_emb)
        return pred, {'topic': topic_emb}
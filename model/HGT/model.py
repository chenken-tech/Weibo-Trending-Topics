import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class HGTModel(nn.Module):
    def __init__(self, graph, input_dim_dict, hidden_dim=64, n_layers=2, n_heads=4, dropout=0.2):
        super(HGTModel, self).__init__()
        self.ntypes = graph.ntypes
        self.etypes = graph.canonical_etypes
        self.num_ntypes = len(self.ntypes)
        self.num_etypes = len(self.etypes)
        self.hidden_dim = hidden_dim

        # 为每种节点类型定义投影层，统一映射到 hidden_dim
        self.projection = nn.ModuleDict({
            ntype: nn.Linear(input_dim_dict[ntype], hidden_dim, bias=True)
            for ntype in input_dim_dict
        })

        # HGT 卷积层
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                dgl.nn.pytorch.conv.HGTConv(
                    in_size=hidden_dim,
                    head_size=hidden_dim // n_heads,
                    num_heads=n_heads,
                    num_ntypes=self.num_ntypes,
                    num_etypes=self.num_etypes,
                    dropout=dropout,
                    use_norm=True
                )
            )

        # 回归 MLP 头
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # 以下为缓存变量，第一次 forward 时构建
        self._node_type_ids = None  # 每个节点的类型索引 (N_total,)
        self._edge_type_ids = None  # 每条边的类型索引 (E_total,)
        self._homo_g = None  # 同构图（只有边索引，无特征）
        self._node_range = {}  # 每种节点类型在全局节点中的起始和结束位置

    def _build_cache(self, g):
        """构建节点类型张量、边类型张量和同构图（仅结构），仅在第一次调用时执行"""
        if self._node_type_ids is not None:
            return

        # 1. 计算每个节点类型在全局节点列表中的偏移量
        node_offsets = {}
        total_nodes = 0
        for ntype in self.ntypes:
            node_offsets[ntype] = total_nodes
            num = g.number_of_nodes(ntype)
            total_nodes += num
            self._node_range[ntype] = (total_nodes - num, total_nodes)

        # 2. 构建节点类型张量
        node_type_list = []
        for ntype in self.ntypes:
            num = g.number_of_nodes(ntype)
            ntype_id = self.ntypes.index(ntype)
            node_type_list.append(torch.full((num,), ntype_id, dtype=torch.long))
        self._node_type_ids = torch.cat(node_type_list, dim=0)

        # 3. 构建边类型张量和同构图的边列表
        src_list = []
        dst_list = []
        edge_type_list = []
        etype_to_id = {etype: idx for idx, etype in enumerate(self.etypes)}

        for stype, etype, dtype in self.etypes:
            subg = g[stype, etype, dtype]
            if subg.number_of_edges() == 0:
                continue
            u, v = subg.edges()
            # 将节点 ID 转换为全局 ID
            src_list.append(u + node_offsets[stype])
            dst_list.append(v + node_offsets[dtype])
            eid = etype_to_id[(stype, etype, dtype)]
            edge_type_list.append(torch.full((len(u),), eid, dtype=torch.long))

        src = torch.cat(src_list)
        dst = torch.cat(dst_list)
        self._homo_g = dgl.graph((src, dst), num_nodes=total_nodes)
        self._edge_type_ids = torch.cat(edge_type_list, dim=0)

    def forward(self, g, device):
        g = g.to(device)
        # 构建缓存（只在第一次 forward 时构建，后续复用）
        self._build_cache(g)

        # 将缓存的张量移动到当前设备
        node_type_ids = self._node_type_ids.to(device)
        edge_type_ids = self._edge_type_ids.to(device)
        homo_g = self._homo_g.to(device)

        # 1. 投影所有节点特征
        all_feats = []
        for ntype in self.ntypes:
            feat = g.nodes[ntype].data['feat'].to(device)
            proj_feat = self.projection[ntype](feat)  # (num_nodes, hidden_dim)
            all_feats.append(proj_feat)
        x = torch.cat(all_feats, dim=0)  # (total_nodes, hidden_dim)

        # 2. 逐层 HGT 卷积
        for layer in self.layers:
            x = layer(homo_g, x, node_type_ids, edge_type_ids)

        # 3. 提取 topic 节点的特征
        start, end = self._node_range['topic']
        topic_features = x[start:end]

        # 4. 回归预测
        pred = self.regressor(topic_features)
        return pred, {'topic': topic_features}
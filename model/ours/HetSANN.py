import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from tqdm import tqdm

from RelationGraphConv import RelationGraphConv
from HeteroConv import HeteroGraphConv


class HetSANNLayer(nn.Module):
    def __init__(self, graph: dgl.DGLHeteroGraph, input_dim: int, hidden_dim: int, n_heads: int = 8,
                 dropout: float = 0.2, residual: bool = True):
        """
        :param graph: a heterogeneous graph
        :param input_dim: int, input dimension
        :param hidden_dim: int, hidden dimension
        :param n_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param residual: boolean, residual connections or not
        """
        super(HetSANNLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.residual = residual

        # ---- 帖子节点关系门控（第二个机制，与 RelationGraphConv 里的熵权注意力并列）----
        # 只用 Linear(D→1)+sigmoid 给每条入边关系算一个**逐节点**权重，
        # 再对这些关系的输出做加权平均。零初始化 → 起点等于原来的等权均值。
        self.gate_ntypes = ("post",)
        self.post_gate = nn.ModuleDict({etype: nn.Linear(input_dim, 1) for etype in graph.etypes})
        for _etype in self.post_gate:
            nn.init.zeros_(self.post_gate[_etype].weight)
            nn.init.zeros_(self.post_gate[_etype].bias)

        # node transformation parameters of each type
        self.node_transformation_weight = nn.ParameterDict({
            ntype: nn.Parameter(torch.randn(input_dim, n_heads * hidden_dim))
            for ntype in graph.ntypes
        })

        # relation attention parameters of each type
        self.relations_attention_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(n_heads, 2 * hidden_dim))
            for etype in graph.etypes
        })

        # hetero conv modules, each RelationGraphConv deals with a single type of relation
        self.hetero_conv = HeteroGraphConv({
            etype: RelationGraphConv(in_feats=(input_dim, input_dim), out_feats=hidden_dim,
                                     num_heads=n_heads, dropout=dropout, negative_slope=0.2)
            for etype in graph.etypes
        })

        if self.residual:
            # residual connection
            self.res_fc = nn.ModuleDict()
            for ntype in graph.ntypes:
                if graph.number_of_dst_nodes(ntype) > 0:
                    self.res_fc[ntype] = nn.Linear(input_dim, n_heads * hidden_dim, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        for weight in self.node_transformation_weight:
            nn.init.xavier_normal_(self.node_transformation_weight[weight], gain=gain)
        for weight in self.relations_attention_weight:
            nn.init.xavier_normal_(self.relations_attention_weight[weight], gain=gain)
        if self.residual:
            for ntype in self.res_fc:
                nn.init.xavier_normal_(self.res_fc[ntype].weight, gain=gain)

    def forward(self, graph: dgl.DGLHeteroGraph, node_features: dict, entropy_weights: dict = None):
        """
        :param graph: dgl.DGLHeteroGraph
        :param node_features: dict, {ntype: node_feature},
        :param entropy_weights: dict, optional, {ntype: tensor of shape (N_ntype, 1)}
                               用于存储每个节点的熵权先验（仅对post类型有效）
        :return: output_features: dict, {relation_type: target_node_features}
        """
        # dictionary of input source features and destination features
        input_src = node_features

        if graph.is_block:
            input_dst = {}
            for ntype in node_features:
                input_dst[ntype] = node_features[ntype][:graph.number_of_dst_nodes(ntype)]
        else:
            input_dst = node_features

        # ========== 新增：构造边级别的熵权先验 ==========
        entropy_prior_dict = {}
        if entropy_weights is not None and 'post' in entropy_weights:
            post_entropy = entropy_weights['post']  # (N_post, 1)
            for stype, etype, dtype in graph.canonical_etypes:
                # 对所有目标节点类型为 post 的关系边，提取对应帖子的熵权
                if dtype == 'post':
                    rel_graph = graph[stype, etype, dtype]
                    if rel_graph.number_of_edges() > 0:
                        # ========== 修复：dst_nodes() -> dstnodes() ==========
                        # DGL 新版本使用 dstnodes() 而不是 dst_nodes()
                        try:
                            # 尝试新版本 API
                            dst_nodes = rel_graph.dstnodes()
                        except AttributeError:
                            # 兼容旧版本 API
                            dst_nodes = rel_graph.dst_nodes()
                        # 获取目标帖子的熵权，shape (E, 1)
                        entropy_prior_dict[etype] = post_entropy[dst_nodes]

        # output_features, dict {(stype, etype, dtype): features}
        relation_features = self.hetero_conv(graph, input_src, input_dst,
                                           self.node_transformation_weight,
                                           self.relations_attention_weight,
                                           entropy_prior_dict=entropy_prior_dict)

        output_features = {}
        # aggregate different types of relations of the target node with sum aggregation
        for ntype in input_dst:
            if graph.number_of_dst_nodes(ntype) != 0:
                rels = [(stype, etype, dtype) for stype, etype, dtype in relation_features
                        if dtype == ntype]
                if not rels:
                    continue
                if ntype in self.gate_ntypes:
                    # 帖子节点：逐节点门控加权平均（替代等权 torch.mean）
                    num, den = None, None
                    for _stype, _etype, _dtype in rels:
                        _f = relation_features[(_stype, _etype, _dtype)]
                        _w = torch.sigmoid(self.post_gate[_etype](_f))
                        num = _f * _w if num is None else num + _f * _w
                        den = _w if den is None else den + _w
                    output_features[ntype] = num / (den + 1e-6)
                else:
                    # 其他节点类型：保持原来的等权平均
                    stacked = torch.stack([relation_features[r] for r in rels], dim=0)
                    output_features[ntype] = torch.mean(stacked, dim=0)

        # residual connection for the target node
        if self.residual:
            for ntype in output_features:
                if ntype in self.res_fc:
                    output_features[ntype] = output_features[ntype] + self.res_fc[ntype](input_dst[ntype])

        # node features after relation crossing layer, {ntype: node_features}
        return output_features


class HetSANN(nn.Module):
    def __init__(self, graph: dgl.DGLHeteroGraph, input_dim_dict: dict, hidden_dim: int, num_layers: int, n_heads: int = 4,
                 dropout: float = 0.2, residual: bool = True):
        """
        :param graph: a heterogeneous graph
        :param input_dim_dict: input dim dictionary
        :param hidden_dim: int, hidden dimension
        :param num_layers: int, number of stacked layers
        :param n_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param residual: boolean, residual connections or not
        """
        super(HetSANN, self).__init__()

        self.input_dim_dict = input_dim_dict
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.residual = residual

        # align the dimension of different types of nodes
        self.projection_layer = nn.ModuleDict({
            ntype: nn.Linear(input_dim_dict[ntype], hidden_dim * n_heads) for ntype in input_dim_dict
        })

        # each layer takes in the heterogeneous graph as input
        self.layers = nn.ModuleList()

        # for each relation_layer
        for layer_num in range(self.num_layers):
            self.layers.append(HetSANNLayer(graph, hidden_dim * n_heads, hidden_dim, n_heads, dropout, residual))

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        for ntype in self.projection_layer:
            nn.init.xavier_normal_(self.projection_layer[ntype].weight, gain=gain)

    def forward(self, blocks: list, node_features: dict, entropy_weights: dict = None):
        """
        :param blocks: list of sampled dgl.DGLHeteroGraph
        :param node_features: node features, dict, {"type": features}
        :param entropy_weights: dict, optional, {ntype: tensor}
        :return:
        """
        # feature projection
        for ntype in node_features:
            node_features[ntype] = self.projection_layer[ntype](node_features[ntype])

        # graph convolution
        for block, layer in zip(blocks, self.layers):
            node_features = layer(block, node_features, entropy_weights=entropy_weights)

        return node_features

    def inference(self, graph: dgl.DGLHeteroGraph, node_features: dict, device: str):
        """
        mini-batch inference of final representation over all node types.
        """
        with torch.no_grad():
            for index, layer in enumerate(self.layers):
                y = {
                    ntype: torch.zeros(
                        graph.number_of_nodes(ntype), self.hidden_dim * self.n_heads) for ntype in graph.ntypes}
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dgl.dataloading.DataLoader(
                    graph,
                    {ntype: torch.arange(graph.number_of_nodes(ntype), dtype=torch.int32).to(device) for ntype in graph.ntypes},
                    sampler,
                    batch_size=1280,
                    shuffle=True,
                    drop_last=False,
                    num_workers=0)

                tqdm_dataloader = tqdm(dataloader, ncols=120)
                for batch, (input_nodes, output_nodes, blocks) in enumerate(tqdm_dataloader):
                    block = blocks[0].to(device)

                    input_features = {ntype: node_features[ntype][input_nodes[ntype].cpu()].to(device) for ntype in input_nodes.keys()}

                    if index == 0:
                        for ntype in input_features:
                            input_features[ntype] = self.projection_layer[ntype](input_features[ntype])

                    h = layer(block, input_features)

                    for k in h.keys():
                        y[k][output_nodes[k]] = h[k].cpu()

                    tqdm_dataloader.set_description(f'inference for the {batch}-th batch in model {index}-th layer')

                node_features = y

        return y
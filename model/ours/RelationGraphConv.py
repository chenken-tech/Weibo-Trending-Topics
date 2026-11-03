import torch
from torch import nn
import torch.nn.functional as F

import dgl
from dgl.ops import edge_softmax
import dgl.function as fn


class RelationGraphConv(nn.Module):

    def __init__(self, in_feats: tuple, out_feats: int, num_heads: int, dropout: float = 0.0,
                 negative_slope: float = 0.2):
        """
        Parameters
        ----------
        in_feats : pair of ints
            Input feature size.
        out_feats : int
            Output feature size.
        num_heads : int
            Number of heads in Multi-Head Attention.
        dropout : float, optional
            Dropout rate, defaults: 0.
        negative_slope : float, optional
            Negative slope rate, defaults: 0.2.
        """
        super(RelationGraphConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = in_feats[0], in_feats[1]
        self._out_feats = out_feats
        self._num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.relu = nn.ReLU()

        # ========== 新增：可学习的熵权缩放参数 ==========
        # 让模型自适应地决定熵权先验的强度
        self.entropy_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, graph: dgl.DGLHeteroGraph, feat: tuple, dst_node_transformation_weight: nn.Parameter,
                src_node_transformation_weight: nn.Parameter, relation_attention_weight: nn.Parameter,
                entropy_prior: torch.Tensor = None):
        """
        Compute graph attention network layer.

        Parameters
        ----------
        graph : specific relational DGLHeteroGraph
        feat : pair of torch.Tensor
            The pair contains two tensors of shape (N_{in}, D_{in_{src}})` and (N_{out}, D_{in_{dst}}).
        dst_node_transformation_weight: Parameter (input_dst_dim, n_heads * hidden_dim)
        src_node_transformation_weight: Parameter (input_src_dim, n_heads * hidden_dim)
        relation_attention_weight: Parameter (n_heads, 2 * hidden_dim)
        entropy_prior: torch.Tensor, optional
            每条边对应的目标帖子熵权先验，shape (E, 1) 或 (E, n_heads, 1)

        Returns
        -------
        torch.Tensor, shape (N, H, D_out)` where H is the number of heads, and D_out is size of output feature.
        """
        graph = graph.local_var()
        # Tensor, (N_src, input_src_dim)
        feat_src = self.dropout(feat[0])
        # Tensor, (N_dst, input_dst_dim)
        feat_dst = self.dropout(feat[1])
        # Tensor, (N_src, n_heads, hidden_dim) -> (N_src, input_src_dim) * (input_src_dim, n_heads * hidden_dim)
        feat_src = torch.matmul(feat_src, src_node_transformation_weight).view(-1, self._num_heads, self._out_feats)
        # Tensor, (N_dst, n_heads, hidden_dim) -> (N_dst, input_dst_dim) * (input_dst_dim, n_heads * hidden_dim)
        feat_dst = torch.matmul(feat_dst, dst_node_transformation_weight).view(-1, self._num_heads, self._out_feats)

        # first decompose the weight vector into [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j, This implementation is much efficient
        # Tensor, (N_dst, n_heads, 1),   (N_dst, n_heads, hidden_dim) * (n_heads, hidden_dim)
        e_dst = (feat_dst * relation_attention_weight[:, :self._out_feats]).sum(dim=-1, keepdim=True)
        # Tensor, (N_src, n_heads, 1),   (N_src, n_heads, hidden_dim) * (n_heads, hidden_dim)
        e_src = (feat_src * relation_attention_weight[:, self._out_feats:]).sum(dim=-1, keepdim=True)
        # (N_src, n_heads, hidden_dim), (N_src, n_heads, 1)
        graph.srcdata.update({'ft': feat_src, 'e_src': e_src})
        # (N_dst, n_heads, 1)
        graph.dstdata.update({'e_dst': e_dst})
        # compute edge attention, e_src and e_dst are a_src * Wh_src and a_dst * Wh_dst respectively.
        graph.apply_edges(fn.u_add_v('e_src', 'e_dst', 'e'))
        # shape (edges_num, heads, 1)
        e = self.leaky_relu(graph.edata.pop('e'))

        # ========== 新增：熵权先验引导注意力 ==========
        if entropy_prior is not None:
            # entropy_prior: (E, 1) 或 (E, n_heads, 1)
            # 将熵权作为注意力偏置，高熵权边获得更高的注意力系数
            if entropy_prior.dim() == 2 and entropy_prior.size(1) == 1:
                # (E, 1) -> (E, n_heads, 1) 广播
                entropy_prior = entropy_prior.unsqueeze(1)  # (E, 1, 1)
            elif entropy_prior.dim() == 1:
                entropy_prior = entropy_prior.unsqueeze(-1).unsqueeze(-1)  # (E, 1, 1)
            # 确保维度匹配
            if entropy_prior.size(1) != self._num_heads:
                # 如果头数不匹配，复制到对应头数
                entropy_prior = entropy_prior.expand(-1, self._num_heads, -1)
            # 用可学习的缩放参数控制熵权的影响强度
            e = e + entropy_prior * torch.abs(self.entropy_scale)

        # compute softmax
        graph.edata['a'] = edge_softmax(graph, e)

        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'msg'), fn.sum('msg', 'ft'))
        # (N_dst, n_heads * hidden_dim), reshape (N_dst, n_heads, hidden_dim)
        dst_features = graph.dstdata.pop('ft').reshape(-1, self._num_heads * self._out_feats)

        dst_features = self.relu(dst_features)

        return dst_features

    @staticmethod
    def gcn_message(edges: dgl.udf.EdgeBatch):
        return {'msg': edges.src['ft'] * edges.data['a']}

    @staticmethod
    def gcn_reduce(nodes: dgl.udf.NodeBatch):
        return {'ft': torch.sum(nodes.mailbox['msg'], 1)}
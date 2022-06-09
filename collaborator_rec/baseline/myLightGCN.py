from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, ModuleList, Linear
from torch.nn.modules.loss import _Loss

from torch_geometric.nn.conv import LGConv
from torch_geometric.typing import Adj, OptTensor


class LightGCN(torch.nn.Module):
    r"""The LightGCN model from the `"LightGCN: Simplifying and Powering
    Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper.
    :class:`~torch_geometric.nn.models.LightGCN` learns embeddings by linearly
    propagating them on the underlying graph, and uses the weighted sum of the
    embeddings learned at all layers as the final embedding
    .. math::
        \textbf{x}_i = \sum_{l=0}^{L} \alpha_l \textbf{x}^{(l)}_i,
    where each layer's embedding is computed as
    .. math::
        \mathbf{x}^{(l+1)}_i = \sum_{j \in \mathcal{N}(i)}
        \frac{1}{\sqrt{\deg(i)\deg(j)}}\mathbf{x}^{(l)}_j.
    Two prediction heads and trainign objectives are provided:
    **link prediction** (via
    :meth:`~torch_geometric.nn.models.LightGCN.link_pred_loss` and
    :meth:`~torch_geometric.nn.models.LightGCN.predict_link`) and
    **recommendation** (via
    :meth:`~torch_geometric.nn.models.LightGCN.recommendation_loss` and
    :meth:`~torch_geometric.nn.models.LightGCN.recommend`).
    .. note::
        Embeddings are propagated according to the graph connectivity specified
        by :obj:`edge_index` while rankings or link probabilities are computed
        according to the edges specified by :obj:`edge_label_index`.
    Args:
        num_nodes (int): The number of nodes in the graph.
        embedding_dim (int): The dimensionality of node embeddings.
        num_layers (int): The number of
            :class:`~torch_geometric.nn.conv.LGConv` layers.
        alpha (float or Tensor, optional): The scalar or vector specifying the
            re-weighting coefficients for aggregating the final embedding.
            If set to :obj:`None`, the uniform initialization of
            :obj:`1 / (num_layers + 1)` is used. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`~torch_geometric.nn.conv.LGConv` layers.
    """
    def __init__(
        self,
        num_node:int,
        num_feat: int,
        embedding_dim: int,
        num_layers: int,
        alpha: Optional[Union[float, Tensor]] = None,
        **kwargs,
    ):
        super().__init__()

        self.num_feat = num_feat
        self.num_node = num_node #old
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        if alpha is None:
            alpha = 1. / (num_layers + 1)

        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)

        #self.embedding = Embedding(num_nodes, embedding_dim)
        self.embedding = Linear(num_feat, embedding_dim)
        self.convs = ModuleList([LGConv(**kwargs) for _ in range(num_layers)])
        self.linklyr = Linear(embedding_dim *2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()

    def get_embedding(self, X:Adj, edge_index: Adj) -> Tensor:
        # x = self.embedding.weight old
        x = self.embedding(X)
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            out = out + x * self.alpha[i + 1]

        return out

    def forward(self, edge_index: Adj, X: Adj,
                edge_label_index: OptTensor = None) -> Tensor:
        r"""Computes rankings for pairs of nodes.
        Args:
            X: the input node features
            edge_index (Tensor or SparseTensor): Edge tensor specifying the
                connectivity of the graph.
            edge_label_index (Tensor, optional): Edge tensor specifying the
                node pairs for which to compute rankings or probabilities.
                If :obj:`edge_label_index` is set to :obj:`None`, all edges in
                :obj:`edge_index` will be used instead. (default: :obj:`None`)
        """
        if edge_label_index is None:
            edge_label_index = edge_index

        #out = torch.mm(X, self.get_embedding(edge_index).T) #total_node_number * embeding_dim 
        out = self.get_embedding(X = X, edge_index= edge_index)
        
        # select rows from features corresponding to the node index 
        out_src = out[edge_label_index[0]]
        out_dst = out[edge_label_index[1]]
        return torch.cat([out_src, out_dst], dim = 1)

    def predict_link(self, X: Adj, edge_index: Adj, edge_label_index: OptTensor = None,
                     prob: bool = False) -> Tensor:
        r"""Predict links between nodes specified in :obj:`edge_label_index`.
        Args:
            prob (bool): Whether probabilities should be returned. (default:
                :obj:`False`)
        """
        #pred = self(edge_index, edge_label_index).sigmoid()
        pred1 = self(edge_index = edge_index, X= X, edge_label_index = edge_label_index) #total events * (embed_dim *2)
        pred = self.linklyr(pred1)
        proba = pred.sigmoid()
        return proba if prob else pred #.round()

    def link_pred_loss(self, pred: Tensor, edge_label: Tensor,
                       **kwargs) -> Tensor:
        r'''Computes the model loss for a link prediction objective via the
        :class:`torch.nn.BCEWithLogitsLoss`.
        Args:
            pred (Tensor): The predictions.
            edge_label (Tensor): The ground-truth edge labels.
            **kwargs (optional): Additional arguments of the underlying
                :class:`torch.nn.BCEWithLogitsLoss` loss function.
        '''
        loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
        return loss_fn(pred, edge_label.to(pred.dtype))

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_feat}, '
                f'{self.embedding_dim}, num_layers={self.num_layers})')



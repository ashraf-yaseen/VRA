"""
model related odule for TGN 
references:
author: [Ginny](jie.zhu@uth.tmc.edu)
"""

import os.path as osp

import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.utils import dropout_adj
from sklearn.metrics import average_precision_score, roc_auc_score

from torch_geometric.datasets import JODIEDataset
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (LastNeighborLoader, IdentityMessage,
                                           LastAggregator)


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc, dropout=0.1, heads =2):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.dropout = dropout 
        self.heads = heads
        self.conv = TransformerConv(in_channels, out_channels // 2, heads= self.heads,
                                    dropout= self.dropout, edge_dim=edge_dim) #test this 

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        # introducing masking some index before feeding
        # in this way, it will not respond to eval() 
        new_edge_index, new_edge_attr = dropout_adj(edge_index,edge_attr,
                                        p= self.dropout,
                                        training= True)
        return self.conv(x, new_edge_index, new_edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, dropout = 0.1):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)
        # Ginny's 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, z_src, z_dst):
        # introducing dropout
        z_src = self.dropout(z_src)
        z_dst = self.dropout(z_dst)
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.tanh()#tanh() or relu()
        return self.lin_final(h)
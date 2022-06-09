"""
models needed for the link prediction
mainly encoder module (graph neural networks) and prediction module (nn)
"""
# general 
import itertools
import numpy as np
import scipy.sparse as sp

# torch 
import torch
import torch.nn as nn
import torch.nn.functional as F

# dgl 
import dgl
from dgl.nn import SAGEConv
import dgl.function as fn


# GCN, encoder 
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, pool = ['gcn','gcn']):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type = pool[0])
        self.conv2 = SAGEConv(h_feats, h_feats, aggregator_type = pool[1])
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    

# decoder/ predictor
class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']        
        
        
# simpler decoder
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]
        
        
        
# whole model 
class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, pool =['gcn','mean']):
        super().__init__()
        self.sage = GraphSAGE(in_features, hidden_features, pool = pool)
        self.pred = MLPPredictor(hidden_features, out_features)
    def forward(self, g, x):
        h = self.sage(g, x)
        return self.pred(g, h) # construct one whole graph for the prediction 

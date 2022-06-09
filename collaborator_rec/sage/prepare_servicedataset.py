"""
refs:
1.https://docs.dgl.ai/tutorials/blitz/6_load_data.html#sphx-glr-tutorials-blitz-6-load-data-py
create a static graph network using the same data for temporal link prediction (preparing in the exact same way) 
create a dataset class
"""
######################################################################
# ``DGLDataset`` Object Overview
# ------------------------------
# 
# Your custom graph dataset should inherit the ``dgl.data.DGLDataset``
# class and implement the following methods:
# 
# -  ``__getitem__(self, i)``: retrieve the ``i``-th example of the
#    dataset. An example often contains a single DGL graph, and
#    occasionally its label.
# -  ``__len__(self)``: the number of examples in the dataset.
# -  ``process(self)``: load and process raw data from disk.

######################################################################
def countUniqueNodes(df, mask):
    """
    if verbose;
    a mini function to print out the data statistics
    """
    temp = df[mask]
    nodes1 = temp.new_author.tolist()
    nodes2 = temp.new_coauthor.tolist()
    node_set = set(nodes1+ nodes2) 
    
    return len(node_set)

import dgl
from dgl.data import DGLDataset
import torch

import pandas as pd
import numpy as np 

import os
import random 
random.seed(2020)

torch.set_default_tensor_type(torch.DoubleTensor)

class CollabDataset(DGLDataset):
    def __init__(self, raw_dir ='service/hulin_wu', verbose = True):
        super().__init__(name='collab', raw_dir = raw_dir, verbose = verbose)

        
    def process(self):
        nodes_data = pd.read_csv(self.raw_dir + 'authors.csv')
        nodes_data.fillna(0., inplace = True)
        node_labels = torch.from_numpy(nodes_data['state_label'].to_numpy()).type(torch.LongTensor) # should be all zeros 
        # needs to go from 4rd column are the node features (by mesh terms of papers)
        node_features = torch.from_numpy(nodes_data.iloc[:,3:].copy().to_numpy())
        # print(type(node_features))
        edges_data = pd.read_csv( self.raw_dir + 'collabs.csv')
        edges_data.sort_values(by=['timestamp'], inplace = True, ignore_index=True)
        # modify this, needs to go from 3rd column are the node features (by mesh terms of papers)
        # we need the appearance of the edges as the weight in prepare_data.py
        edges_features = torch.from_numpy(edges_data['weight'].to_numpy())
        edges_src = torch.from_numpy(edges_data['new_author'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['new_coauthor'].to_numpy())
        edges_time = torch.from_numpy(edges_data['timestamp'].to_numpy())
        #edges_label = torch.ones(len(edges_time), dtype= torch.long)
        # node_features, edges_features = node_features.type(torch.DoubleTensor), edges_features.type(torch.DoubleTensor)

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['weight'] = edges_features# we could use the appearance of links as the weights
        self.graph.edata['time'] = edges_time
        #self.gragh.edata['label'] = edges_label
 

        # split processing
        train_mask = (edges_data['mask'].to_numpy() == 0)
        val_mask = (edges_data['mask'].to_numpy() == 1)
        test_mask = (edges_data['mask'].to_numpy() == 2)

        ## train, validation and test
        self.graph.edata['train_mask'] = torch.from_numpy(train_mask)
        self.graph.edata['val_mask'] = torch.from_numpy(val_mask)
        self.graph.edata['test_mask'] = torch.from_numpy(test_mask)


        if self.verbose:
            # we are calculating something new
            n_nodes = nodes_data.shape[0]
            n_edges = edges_data.shape[0]
            n_train_nodes = countUniqueNodes(edges_data, train_mask)
            n_valid_nodes = countUniqueNodes(edges_data, val_mask)
            n_test_nodes = countUniqueNodes(edges_data, test_mask)
            print("The dataset has {} interactions, involving {} different nodes".format(n_edges,
                                                                      n_nodes))
            print("The training dataset has {} interactions, involving {} different nodes".format(
            train_mask.sum().item(), n_train_nodes))
            print("The validation dataset has {} interactions, involving {} different nodes".format(
            val_mask.sum().item(), n_valid_nodes))
            print("The test dataset has {} interactions, involving {} different nodes".format(
            test_mask.sum().item(), n_test_nodes))


        
    def __getitem__(self, i):
        return self.graph
    
    def __len__(self):
        return 1
    

         
    
"""
def main()    
    dataset = CollabDataset()
    graph = dataset[0]
    print(graph)

if __name__ == "__main__":
    main()    
"""

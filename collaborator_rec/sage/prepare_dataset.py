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
    def __init__(self, raw_dir = 'sage/data/20192020/', verbose = True):
        super().__init__(name='collab', raw_dir = raw_dir, verbose = verbose)

        
    def process(self):
        nodes_data = pd.read_csv(self.raw_dir + 'authors.csv')
        node_labels = torch.from_numpy(nodes_data['state_label'].to_numpy()).type(torch.LongTensor) # should be all zeros 
        # needs to go from 4rd column are the node features (by mesh terms of papers)
        node_features = torch.from_numpy(nodes_data.iloc[:,3:].copy().to_numpy())
        # print(type(node_features))
        edges_data = pd.read_csv( self.raw_dir +'collabs.csv')
        edges_data.sort_values(by=['timestamp'], inplace = True, ignore_index=True)
        # modify this, needs to go from 3rd column are the node features (by mesh terms of papers)
        # we need the appearance of the edges as the weight in prepare_data.py
        edges_features = torch.from_numpy(edges_data['weight'].to_numpy())
        edges_src = torch.from_numpy(edges_data['new_author'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['new_coauthor'].to_numpy())
        edges_time = torch.from_numpy(edges_data['timestamp'].to_numpy())
        edges_label = torch.ones(len(edges_time), dtype= torch.long)
        # node_features, edges_features = node_features.type(torch.DoubleTensor), edges_features.type(torch.DoubleTensor)

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['weight'] = edges_features# we could use the appearance of links as the weights
        self.graph.edata['time'] = edges_time
        #self.gragh.edata['label'] = edges_label
 

        # split processing
        timestamps = edges_data['timestamp'].values
        val_time, test_time = list(np.quantile(edges_data['timestamp'].values, [0.70, 0.85]))
        train_mask = (timestamps <= val_time)

        # more stuff for comprehensiveness 
        train = edges_data[train_mask]
        nontrain = edges_data[~train_mask]

        n_nodes = nodes_data.shape[0] 
        node_set = set(nodes_data.id.tolist())
        print('before processing, total nodes :{}'.format(len(node_set)))
        # node counts get the test first  
        test_node_set = set(nontrain.new_author.tolist() + nontrain.new_coauthor.tolist())
        # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
        # their edges from training
        print('before processing, total none training nodes :{}'.format(len(test_node_set)))

        # test set
        new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_nodes)))
        new_test_mask = edges_data.new_author.map(lambda x: x in new_test_node_set).values
        new_test_mask2 = edges_data.new_coauthor.map(lambda x: x in new_test_node_set).values    

        # Mask which is true for edges with both destination and source not being new test nodes (because
        # we want to remove all edges involving any new test node)
        observed_edges_mask = ((~new_test_mask) & (~new_test_mask2))

        # For train we keep edges happening before the validation time which do not involve any new node
        # used for inductiveness
        train_mask = ((timestamps <= val_time) & observed_edges_mask)

        # define the new nodes sets for testing inductiveness of the model
        train_node_set = set(train.new_author.tolist() + train.new_coauthor.tolist())
        print('should be zero: {}'.format(len(train_node_set & new_test_node_set)))
        # assert len(train_node_set & new_test_node_set) == 0
        new_node_set = node_set - train_node_set

        val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
        test_mask = (timestamps > test_time)

        edge_contains_new_node_mask = np.array(
          [(a in new_node_set or b in new_node_set) for a, b in \
           zip(edges_data['new_author'].values, edges_data['new_coauthor'].values)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

        ## train, validation and test
        self.graph.edata['train_mask'] = torch.from_numpy(train_mask)
        self.graph.edata['val_mask'] = torch.from_numpy(val_mask)
        self.graph.edata['test_mask'] = torch.from_numpy(test_mask)

        ## validation and test with edges that at least has one new node (not in training set)
        self.graph.edata['new_node_val_mask'] = torch.from_numpy(new_node_val_mask)
        self.graph.edata['new_node_test_mask'] = torch.from_numpy(new_node_test_mask)


        if self.verbose:
            # we are calculating something new
            n_edges = edges_data.shape[0]
            n_train_nodes = countUniqueNodes(edges_data, train_mask)
            n_valid_nodes = countUniqueNodes(edges_data, val_mask)
            n_test_nodes = countUniqueNodes(edges_data, test_mask)
            n_new_node_valid = countUniqueNodes(edges_data, new_node_val_mask)
            n_new_node_test = countUniqueNodes(edges_data, new_node_test_mask)
            print("The dataset has {} interactions, involving {} different nodes".format(n_edges,
                                                                      n_nodes))
            print("The training dataset has {} interactions, involving {} different nodes".format(
            train_mask.sum().item(), n_train_nodes))
            print("The validation dataset has {} interactions, involving {} different nodes".format(
            val_mask.sum().item(), n_valid_nodes))
            print("The test dataset has {} interactions, involving {} different nodes".format(
            test_mask.sum().item(), n_test_nodes))
            print("The new node validation dataset has {} interactions, involving {} different nodes".format(
            new_node_val_mask.sum().item(), n_new_node_valid))
            print("The new node test dataset has {} interactions, involving {} different nodes".format(
            new_node_test_mask.sum().item(), n_new_node_test))
            print("{} nodes were used for the inductive valid + test; i.e. are never seen during training".format(
            len(new_test_node_set)))      


        
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

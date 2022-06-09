# general 
import logging
from pathlib import Path
import time 
import itertools
from itertools import combinations, product
import numpy as np
import pandas as pd 
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score
# import pickle
import gc
import json
import random
import re
import os 
import fnmatch


# torch 
import torch
import torch.nn as nn
import torch.nn.functional as F

# dgl 
import dgl
from dgl.nn import SAGEConv
import dgl.function as fn
from dgl.data.utils import save_graphs, load_graphs
"""
refs: https://docs.dgl.ai/tutorials/blitz/4_link_predict.html
util funcs needed during model loading, training and predictions
"""

np.random.seed(2021)


### data related
## old and useless 
def split_edges(graph, newNodeMask = True):
    
    # sources and destinations of links 
    u, v = graph.edges()
    
    # training, validation and test
    train_pos_u, train_pos_v = u[graph.edata['train_mask'].type(torch.bool)], v[graph.edata['train_mask'].type(torch.bool)]
    val_pos_u, val_pos_v = u[graph.edata['val_mask'].type(torch.bool)], v[graph.edata['val_mask'].type(torch.bool)]
    test_pos_u, test_pos_v = u[graph.edata['test_mask'].type(torch.bool)], v[graph.edata['test_mask'].type(torch.bool)]

    # Find all negative edges
    # we need to take care of both on cpu or GPU
    if graph.device == torch.device('cpu'):
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
        
    else:
        # we need to have the GPU tensors here 
        adj = sp.coo_matrix((np.ones(len(u)), (u.cpu().numpy(), v.cpu().numpy())))
    
    adj_neg = 1 - adj.todense() - np.eye(graph.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    # get the negative samples for train, validation and test
    neg_eids = np.random.choice(len(neg_u), graph.number_of_edges())
    
    train_neg_u, train_neg_v = neg_u[neg_eids[graph.edata['train_mask'].cpu().numpy()]], \
    neg_v[neg_eids[graph.edata['train_mask'].cpu().numpy()]]
    val_neg_u, val_neg_v = neg_u[neg_eids[graph.edata['val_mask'].cpu().numpy()]], \
    neg_v[neg_eids[graph.edata['val_mask'].cpu().numpy()]]
    test_neg_u, test_neg_v = neg_u[neg_eids[graph.edata['test_mask'].cpu().numpy()]], \
    neg_v[neg_eids[graph.edata['test_mask'].cpu().numpy()]]
    
    outputs = (train_pos_u, train_pos_v, train_neg_u, train_neg_v, \
               val_pos_u, val_pos_v, val_neg_u, val_neg_v, \
               test_pos_u, test_pos_v, test_neg_u, test_neg_v)
    
    if newNodeMask:
        new_test_pos_u, new_test_pos_v = u[graph.edata['new_node_test_mask'].type(torch.bool)], \
        v[graph.edata['new_node_test_mask'].type(torch.bool)]
        new_val_pos_u, new_val_pos_v = u[graph.edata['new_node_val_mask'].type(torch.bool)],\
        v[graph.edata['new_node_val_mask'].type(torch.bool)]

        new_test_neg_u, new_test_neg_v = neg_u[neg_eids[graph.edata['new_node_test_mask'].cpu().numpy()]], \
        neg_v[neg_eids[graph.edata['new_node_test_mask'].cpu().numpy()]]
        new_val_neg_u, new_val_neg_v = neg_u[neg_eids[graph.edata['new_node_val_mask'].cpu().numpy()]], \
        neg_v[neg_eids[graph.edata['new_node_val_mask'].cpu().numpy()]]
        
        
        outputs = outputs + (new_val_pos_u, new_val_pos_v, new_val_neg_u, new_val_neg_v, \
                             new_test_pos_u, new_test_pos_v, new_test_neg_u, new_test_neg_v)
    
    return outputs 


def construct_wEdges(graph, outputs):
    """
    construct graphs using train, valid and test links from outputs of split_edges()
    if outputs is longer, then it has new_node validation/test as well
    outputs: should be 12 or 20-element tuple from split edges
    
    """
    eids = np.arange(graph.number_of_edges())
    train_g = dgl.remove_edges(graph, np.concatenate([eids[graph.edata['val_mask'].cpu().numpy()], \
                                                      eids[graph.edata['test_mask'].cpu().numpy()]])) #neew to supply 

    if len(outputs) == 12:
        train_pos_u, train_pos_v, train_neg_u, train_neg_v, \
               val_pos_u, val_pos_v, val_neg_u, val_neg_v, \
               test_pos_u, test_pos_v, test_neg_u, test_neg_v = outputs 
        
        train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=graph.number_of_nodes())
        train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=graph.number_of_nodes()).to(graph.device)

        test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=graph.number_of_nodes())
        test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=graph.number_of_nodes()).to(graph.device)


        val_pos_g = dgl.graph((val_pos_u, val_pos_v), num_nodes=graph.number_of_nodes())
        val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=graph.number_of_nodes()).to(graph.device)
        
        
        outgs =  (train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g)
        
    if len(outputs) == 20:
        train_pos_u, train_pos_v, train_neg_u, train_neg_v, \
               val_pos_u, val_pos_v, val_neg_u, val_neg_v, \
               test_pos_u, test_pos_v, test_neg_u, test_neg_v,\
               new_val_pos_u, new_val_pos_v, new_val_neg_u, new_val_neg_v, \
               new_test_pos_u, new_test_pos_v, new_test_neg_u, new_test_neg_v = outputs
        
        # all links 
        train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=graph.number_of_nodes())
        train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=graph.number_of_nodes()).to(graph.device)

        test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=graph.number_of_nodes())
        test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=graph.number_of_nodes()).to(graph.device)

        val_pos_g = dgl.graph((val_pos_u, val_pos_v), num_nodes=graph.number_of_nodes())
        val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=graph.number_of_nodes()).to(graph.device)
        
        # involving new nodes only
        new_test_pos_g = dgl.graph((new_test_pos_u, new_test_pos_v), num_nodes=graph.number_of_nodes())
        new_test_neg_g = dgl.graph((new_test_neg_u, new_test_neg_v), num_nodes=graph.number_of_nodes()).to(graph.device)
        
        new_val_pos_g = dgl.graph((new_val_pos_u, new_val_pos_v), num_nodes=graph.number_of_nodes())
        new_val_neg_g = dgl.graph((new_val_neg_u, new_val_neg_v), num_nodes=graph.number_of_nodes()).to(graph.device)
        
        outgs = (train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g, \
                 new_val_pos_g, new_val_neg_g, new_test_pos_g,  new_test_neg_g)

    else:
        print('please double check your inputs!')
    
    return outgs



def construct_negEdges(graph, k, newNode = True, service = False):
    
    src, dst = graph.edges()
    
    if not service:
        neg_src = src.repeat_interleave(k)
        neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,)).to(neg_src.device)

        # add the edges to the orignal graph 
        if newNode:
            graph.add_edges(neg_src, neg_dst, {'label': torch.zeros(len(neg_src), dtype = torch.long).to(neg_src.device), \
                                       'train_mask': graph.edata['train_mask'].repeat_interleave(k) , \
                                       'val_mask': graph.edata['val_mask'].repeat_interleave(k), \
                                       'test_mask': graph.edata['test_mask'].repeat_interleave(k),\
                                       'new_node_val_mask': graph.edata['new_node_val_mask'].repeat_interleave(k),\
                                       'new_node_test_mask':graph.edata['new_node_test_mask'].repeat_interleave(k)})
        else:
            graph.add_edges(neg_src, neg_dst, {'label': torch.zeros(len(neg_src), dtype = torch.long).to(neg_src.device), \
                               'train_mask': graph.edata['train_mask'].repeat_interleave(k) , \
                               'val_mask': graph.edata['val_mask'].repeat_interleave(k), \
                               'test_mask': graph.edata['test_mask'].repeat_interleave(k),})
    else:
        n = graph.edata['train_mask'].sum() + graph.edata['val_mask'].sum() 
        nk = n *k 
        neg_src = src[:n].repeat_interleave(k)
        neg_dst = torch.randint(0, graph.num_nodes(), (nk,)).to(neg_src.device)

        graph.add_edges(neg_src, neg_dst, {'label': torch.zeros(len(neg_src), dtype = torch.long).to(neg_src.device), \
                               'train_mask': graph.edata['train_mask'][:n].repeat_interleave(k), \
                               'val_mask': graph.edata['val_mask'][:n].repeat_interleave(k), \
                               'test_mask': torch.zeros_like(graph.edata['val_mask'][:n].repeat_interleave(k)).to(neg_src.device)})
        
    return graph




def inductive_edge_split(g, newNode = True):
    """
    a mini function to split the graph based on train, val and test mask 
    """
    train_g = dgl.edge_subgraph(g, g.edata['train_mask'].bool(),  relabel_nodes = False, store_ids = True)
    val_g = dgl.edge_subgraph(g, g.edata['val_mask'].bool(), relabel_nodes = False, store_ids = True)
    test_g = dgl.edge_subgraph(g, g.edata['test_mask'].bool(), relabel_nodes = False, store_ids = True)
    # we need to random shuffle order, others it will learn first halfs are 1s and the second half are 0s 
    if newNode:
        new_val_g = dgl.edge_subgraph(g, g.edata['new_node_val_mask'].bool(), relabel_nodes = False, store_ids = True)
        new_test_g = dgl.edge_subgraph(g, g.edata['new_node_test_mask'].bool(), relabel_nodes = False, store_ids = True)

        return train_g, val_g, test_g, new_val_g, new_test_g
    else:
        return  train_g, val_g, test_g
    

def feat_labels(g):
    feat = g.ndata['feat']
    label = g.edata['label']
    return feat, label #.cpu().numpy()
    
    
    
#### training related  
def create_log(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path(args.save_path +"log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler( args.save_path + 'log/{}.log'.format(str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)
    
    return logger



def tiny_train(model, g, feats, device, y, noy = False):
    pred = model(g.to(device), feats.to(device))
    p = F.softmax(pred, dim =1)
    prob1 = p[:,1].detach().cpu().numpy()
    if not noy:
        auc = roc_auc_score(y.cpu().numpy(), prob1)
        ap = average_precision_score(y.cpu().numpy(),prob1)
        return  model, pred, auc, ap
    else:
        return model, pred 



def train_epochs(logger, epochs, model, train_g, train_feats, train_y, \
                 val_g, val_feats, val_y, \
                 new_val_g, new_val_feats, new_val_y,\
                 device, opt, loss_fcn, path = 'service/hulin_wu/sage', every = 5, newNode = True):
    
    best_auc, nn_best_auc = 0., 0.
    best_ap, nn_best_ap = 0., 0.

    model.to(device)
    train_y = train_y.to(device)
    for epoch in range(epochs):
        
        start_epoch = time.time()
        model.train()
        model, pred, auc, ap = tiny_train(model, train_g, train_feats, device, train_y)
        loss = loss_fcn(pred, train_y.long().to(device))

        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())
        print('training auc = {},  training ap = {}'.format(auc, ap))

        with torch.no_grad():
            model.eval()
            model, pred_, val_auc, val_ap = tiny_train(model, val_g, val_feats, \
                                                       device, val_y)
            if newNode:
                model, pred_, new_val_auc, new_val_ap = tiny_train(model, new_val_g, new_val_feats, \
                                                                  device, new_val_y)

        epoch_time = time.time() - start_epoch

        logger.info('epoch: {} took {:.2f}s'.format(epoch, epoch_time))
        logger.info('Epoch loss: {}'.format(loss.item()))
        if newNode:
            logger.info(
              'val auc: {}, new node val auc: {}'.format(val_auc, new_val_auc))
            logger.info(
              'val ap: {}, new node val ap: {}'.format(val_ap, new_val_ap))
        else: 
            logger.info(
              'val auc: {}'.format(val_auc))
            logger.info(
              'val ap: {}'.format(val_ap))


        if val_ap > best_ap:
            best_ap = val_ap
            best_auc = val_auc
            if newNode:
                nn_best_ap = new_val_ap
                nn_best_auc = new_val_auc
            torch.save(model, path + 'model.pt')

        if epoch % every == 0:
            if newNode:
                print('In epoch {}, loss: {}, val AUC: {}, new nodes val AUC: {}'.format(epoch, loss.item(), val_auc, new_val_auc))
                print('val AP: {}, new nodes val AP: {}'.format(val_ap, new_val_ap))
            else:
                print('In epoch {}, loss: {}, val AUC: {}, val AP: {}'.format(epoch, loss.item(), val_auc, val_ap))

    if nn_best_auc > 0.0:
        logger.info('best val AP found: {}, with new nodes AP: {}'.format(best_ap, nn_best_ap)) 
        logger.info('best val AUC found: {}, with new nodes AUC: {}'.format(best_auc, nn_best_auc)) 
    else:
        logger.info('best val AP found: {}'.format(best_ap)) 
        logger.info('best val AUC found: {}'.format(best_auc)) 
    model = torch.load(path + 'model.pt')

    return model




def predictions(logger, model, \
                test_g, test_feats, test_y, new_test_g, new_test_feats, new_test_y,\
                device, path = 'sage/20192020/', newNode = True):
    with torch.no_grad():
        model.eval()
        model, pred_, test_auc, test_ap = tiny_train(model, test_g, test_feats,\
                                                     device, test_y)
        # save results
        torch.save({'pred': pred_, 'labels':test_y}, path + 'test_pred.pkl')
        
        if newNode:
            model, new_pred_, new_test_auc, new_test_ap = tiny_train(model, new_test_g, new_test_feats, \
                                                                   device, new_test_y)
            torch.save({'pred': new_pred_, 'labels':new_test_y}, path +'nn_test_pred.pkl')

            logger.info('test  AUC: {}, new nodes test AUC: {}'.format( test_auc, new_test_auc))
            logger.info('test AP: {}, new nodes test AP: {}'.format(test_ap, new_test_ap))
        else:
            logger.info('test  AUC: {}'.format( test_auc))
            logger.info('test AP: {}'.format(test_ap))
    logging.shutdown()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
    

def recommend(logger, model, \
                test_g, test_feats, test_y,\
                device, author_dict, firstk = 30, path = 'service/hulin_wu/', \
                f_name = 'hulin', l_name = 'wu', m_name = ''):
    with torch.no_grad():
        model.eval()
        model, pred_ = tiny_train(model, test_g, test_feats,\
                                                     device, test_y, noy = True)
    # save results
    pred = pred_[:, 1].cpu().numpy() # take the second columns
    idx = np.argsort(-pred)[:firstk]
    _, dst = test_g.edges()
    author_ids = np.take(dst.cpu().numpy(),idx)
    
    # result should be names, and their 5 article links
    result = {} 
    ls =[] 
    for id in author_ids:
        tempname = author_dict[id]['name']
        n = len(author_dict[id]['pmid_ls'])
        i = random.sample(list(np.arange(n)), 1)[0]
        tempname_ls = tempname.split(' ')[::-1] #reverse to lm,f
        temp = {'collaborator': tempname,
                'pubMed articles link': 'https://pubmed.ncbi.nlm.nih.gov/?term='+ tempname_ls[0] + tempname_ls[1]\
                + '&cauthor_id=' + author_dict[id]['pmid_ls'][i]} #introduce a bit of variation
        ls.append(temp)
    result['recommended'] = ls 
    name = f_name+ m_name + l_name
    with open(path + name + '_newresult.json', 'w') as fp:
        json.dump(result, fp)
    
    logger.info('produced {} recommended collaborators for {} '.format(firstk, name))    
    logging.shutdown()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
        
        
## find files case insensitive         
def findfiles(which, where='.'):
    '''Returns list of filenames from `where` path matched by 'which'
       shell pattern. Matching is case-insensitive.'''
    
    # TODO: recursive param with walk() filtering
    rule = re.compile(fnmatch.translate(which), re.IGNORECASE)
    return [name for name in os.listdir(where) if rule.match(name)]
        
        

## old and useless 

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(pos_score.device)
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_scores(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu(). numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def train_batch(model, predmodel, train_g, train_pos_g, train_neg_g, optimizer, device):
   
    model.train()
    predmodel.train()
    h = model(train_g, train_g.ndata['feat'])
    pos_score = predmodel(train_pos_g, h)
    neg_score = predmodel(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return h, model, predmodel, optimizer, loss


def train_epochs_old(model, predmodel, train_g, train_pos_g, train_neg_g, optimizer, device, val_pos_g, val_neg_g, \
                 epochs = 100, every =5, path = 'sage/', new_val_pos_g = None, new_val_neg_g = None ):
    
    # let's use AP instead, to be consistent with the tgn
    best_auc, nn_best_auc = 0., 0.
    best_ap, nn_best_ap = 0., 0.
    for e in range(epochs):
        gc.collect()
        torch.cuda.empty_cache()

        h, model, predmodel, optimizer, loss = train_batch(model, predmodel, train_g, train_pos_g, train_neg_g, optimizer, device)
        with torch.no_grad():
            
            pos_score = predmodel(val_pos_g, h)
            neg_score = predmodel(val_neg_g, h)

            if new_val_pos_g and new_val_neg_g:
                new_pos_score = predmodel(new_val_pos_g, h)
                new_neg_score = predmodel(new_val_neg_g, h)

        if e % every == 0:
            val_auc, val_ap = compute_scores(pos_score, neg_score)
            if val_auc > best_auc:
                best_ap = val_ap
                best_auc = val_auc
                torch.save(model, path + 'model.pt')
                torch.save(predmodel, path + 'predmodel.pt')
                if new_val_pos_g and new_val_neg_g:
                    new_val_auc, new_val_ap = compute_scores(new_pos_score, new_neg_score)
                    nn_best_auc = new_val_auc
                    nn_best_ap = new_val_ap
                    print('In epoch {}, loss: {}, val AUC: {}, new nodes val AUC: {}'.format(e, loss, val_auc, new_val_auc))
                    print('val AP: {}, new nodes val AP: {}'.format(val_ap, new_val_ap))
                else:
                    print('In epoch {}, loss: {}, val AUC: {}, val AP'.format(e, loss, val_auc, val_ap))
              
            '''
            if val_auc > best_auc:
                best_auc = val_auc
                if new_val_auc:
                    nn_best_auc = new_val_auc
                    print('In epoch {}, loss: {}, val AUC: {}, new nodes val AUC: {}'.format(e, loss, val_auc, new_val_auc))
                else:
                    print('In epoch {}, loss: {}, val AUC: {}'.format(e, loss, val_auc))'''
            
    if nn_best_auc > 0.0:
        print('best val AP found: {}, with new nodes AP: {}'.format(best_ap, nn_best_ap)) 
        print('best val AUC found: {}, with new nodes AUC: {}'.format(best_auc, nn_best_auc)) 
    else:
        print('best val AP found: {}'.format(best_ap))
        print('best val AUC found: {}'.format(best_auc))
    return h, model, predmodel   
    


"""
This acts as fore-runner before prepare_dataset.py
after running this file, we will have data in tow separate csv files 
1. authors.csv that has ids to corresponding authors, state_label, and finally (and authors/node features as represented by their published articles mesh? (in using all) state_labels == 0: df_mapping
2. collabs.csv that has author, coauthor pairs,  and time (publcation year+month+day), weight: how many times that the recurrent links appears (using all info) & 'mask' (engineered mask for testing purpose): df_long 
"""

import torch
import pickle
import pandas as pd
from torch_geometric.data import Data
import networkx as nx
from scipy.sparse import coo_matrix
from itertools import combinations, product
from collections import defaultdict

import math
import random
import os.path as osp
from itertools import chain
import sys
from ast import literal_eval

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.csgraph import shortest_path
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.nn import ModuleList, Linear, Conv1d, MaxPool1d

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges, k_hop_subgraph,
                                   to_scipy_sparse_matrix)

class yearly_authors():
    def __init__(self, authfile, service_dict, savepath,\
                 l_name = 'wu', f_name = 'hulin', m_name = '', options = 'mesh', val_ratio = 0.2,
                 exclude_users = ''):  
        self.file = pickle.load(open(authfile, 'rb'))
        self.service_dict = service_dict # directly from the service.scrap_pubs()
        # year to extract the graph from
        # self.year = years
        self.l_name = l_name
        self.f_name = f_name
        self.m_name = m_name
        if self.m_name.strip() == '':
            self.name = self.f_name + ' '  + self.l_name 
        else:
            self.name = self.f_name + ' ' +  self.m_name +  ' ' + self.l_name 
        self.val_ratio = val_ratio
        self.maxLen = 50000 #maximum of pubmed articles we will get, for first one 40000
        self.savepath = savepath
        self.edge_indx_coo = None
        self.hasNodeLabel = False
        self.df_thin = None
        self.exclude_users = exclude_users
        self.df_authors()
        self.graphize()
        self.options = options
        self.feature_x(options=self.options)
        
    def df_authors(self, date_col = 'pubdate'):
        """
        sorting authors' publications by year,
        processing the author list into coupled pair
        processing the format into a list of 
        save mapping pickles & graphs
        """
        # merge with exisiting data
        self.file = dict(list(self.file.items()) + list(self.service_dict.items()))
        # sort by years first
        self.file_sorted = {k: v for k, v in sorted(self.file.items(), key=lambda item: str(item[1][date_col]))}
        self.file_year = self.file_sorted 
        # check the length, if too many, cut
        if len(self.file_year) > self.maxLen:
            # get the last
            self.file_year = {k: self.file_year[k] for k in list(self.file_year)[-self.maxLen:]}
        with open(self.savepath  + 'data.pickle', 'wb') as handle:
            pickle.dump(self.file_year, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.df = pd.DataFrame.from_dict(self.file_year, orient='index')
        # do the timestamp
        self.df['timestamp'] = self.df[date_col].str.split(pat='-', expand = True).iloc[:,0].copy()
        ## or we might be able to incorporate more precise time!"""
        self.df['timestamp2'] = self.df[date_col].str.split(pat='-', expand = True).iloc[:,1].copy().str.rjust(2, '0')
        self.df['timestamp2'] = self.df['timestamp2'].fillna('00')
        self.df['timestamp3'] = self.df[date_col].str.split(pat='-', expand = True).iloc[:,2].copy().str.rjust(2, '0')
        self.df['timestamp3'] = self.df['timestamp3'].fillna('00')
        self.df['timestamp'] = self.df['timestamp'].copy() + self.df['timestamp2'] + self.df['timestamp3']

        self.df['timestamp'] = self.df['timestamp'].astype(int)
        self.df.sort_values(by=['timestamp'], inplace = True, ignore_index=True)
        self.df_thin = self.df[['authors', 'pmid', 'mesh_terms', 'journal', 'affiliations', 'link', 'timestamp']]
        # mesh to pmid
        self.mesh_dict = dict(zip(self.df_thin['pmid'].tolist(), self.df_thin['mesh_terms'].tolist()))
        # we need to save a authors to pubmed links dict as well
        
        # combinations of tuples
        authortuple = []
        time = []
        pmid = []
        # affis = []
        for idx, row in self.df_thin.iterrows():
            #temp = row['authors']
            temp = list(combinations(row['authors'].strip('').rsplit(';'), 2))
            authortuple.extend(temp)
            rep = len(temp)
            time.extend([row['timestamp']]* rep)
            pmid.extend([row['pmid']]* rep)
            # affis.extend([row['affiliations']]* rep)
        self.df_long = pd.DataFrame({'authors': authortuple, 'timestamp': time, 'pmid': pmid})#,'affiliations':affis}) most empty
        self.df_long[['author', 'coauthor']] = pd.DataFrame(self.df_long['authors'].tolist(), index= self.df_long.index)
        self.df_long[['author', 'coauthor']] = self.df_long[['author', 'coauthor']].apply(lambda x: x.str.strip())
        self.df_long.reset_index(drop=True, inplace= True)
      
    
    def graphize(self):
        
        """
        processing the dictionary into what SAGE needs
        get mapping
        get graph 

        """
        # create mapping
        people = list(set(self.df_long.author.unique().tolist() + self.df_long.coauthor.unique().tolist()))
        self.num_nodes = len(people)
  
        le = LabelEncoder()
        ids = le.fit_transform(people)
        self.mapdict = dict(zip(le.classes_, range(len(le.classes_))))
        with open(self.savepath  + 'mapdict.pickle', 'wb') as handle:
            pickle.dump(self.mapdict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.names = le.classes_

        self.mapping = pd.DataFrame(list(zip(le.classes_, range(len(le.classes_)))),
               columns =['authors', 'id'])
        self.mapping['state_label']= pd.Series([0]*self.mapping.shape[0]) 
        
        # mapping for author and coauthor
        def people2idx(key):
            return self.mapdict[key]
        self.df_long['new_author'] = self.df_long['author'].apply(people2idx)
        self.df_long['new_coauthor'] = self.df_long['coauthor'].apply(people2idx)
        # need weight of the edges:
        match = pd.DataFrame(self.df_long.groupby(['author','coauthor']).size().reset_index().rename(columns={0:'weight'}))
        self.df_long = self.df_long.merge(match, how='left', on=['author','coauthor'])
        self.df_long = self.df_long[['new_author', 'new_coauthor', 'timestamp', 'pmid', 'weight']]
        
        # for later recommendation to produce more human-friendly result showcase
        pubs1 = self.df_long.groupby('new_author')['pmid'].agg(lambda x: list(set(x))).reset_index(name='part1')
        dict1 = pd.Series(pubs1.part1.values, index = pubs1.new_author).to_dict()
        pubs2 = self.df_long.groupby('new_coauthor')['pmid'].agg(lambda x: list(set(x))).reset_index(name='part2')
        dict2 = pd.Series(pubs2.part2.values, index = pubs2.new_coauthor).to_dict()
        for k, v in dict2.items():
            if k in dict1:
                dict1[k].extend(v)
            else:
                dict1[k] = v
        self.authorID2Pubs = dict1
        # new dictionary with k as the author id, then a nested dict of 'pmid_ls' and 'name' as the value
        author_refs = {}
        for k, v in self.authorID2Pubs.items():
            get_name = self.names[int(k)]
            try:
                v.remove('999999')
            except:
                pass
            author_refs[k] = {'name':get_name, 'pmid_ls': v}
        with open(self.savepath+'author_refs.pickle', 'wb') as handle:
            pickle.dump(author_refs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        # we need to add 'train, valid and test mask here as well: as the 'mask' column with value [0, 1, 2]
        n_train =  int(self.df_long.shape[0] * (1- self.val_ratio))
        n_val =  self.df_long.shape[0] - n_train
        self.df_long['mask'] = pd.Series([0]*n_train + [1]*n_val) 
        
        # deliberately create testing pairs:
        exist = [v['authors'] for i, v in self.service_dict.items()]
        exist2 = [x.split(';') for x in exist]
        exist_users = [item for sublist in exist2 for item in sublist]
        possible = list(set(people).difference(set(self.exclude_users), set(exist_users)))
        possible_ids = [self.mapdict[e] for e in possible]

        ## double check this name, users's actual name in the publication
        serve_id = self.mapdict[self.name] #make sure they appear as exactly they are in the pubMed
        new_author = [serve_id]* len(possible_ids)
        new_coauthor = possible_ids
        timestamp = [self.df_long['timestamp'].iloc[-1] + 1] * len(possible_ids)
        pmid = ['999999']* len(possible_ids)
        weight = [0] * len(possible_ids)
        mask = [2] * len(possible_ids)
        
        df_test = pd.DataFrame({'new_author': new_author, 'new_coauthor': new_coauthor,\
                               'timestamp': timestamp, 'pmid': pmid, 'weight': weight, 'mask': mask})
        self.df_long = pd.concat([self.df_long, df_test], ignore_index = True)
        self.df_long.to_csv(self.savepath + 'collabs.csv', index = False)
        

        

        
    def feature_x(self, options='mesh'):
        """
        get node features, if we want to use mesh terms or other content (pubs) to represnet the authors
        all using tfidf, max_features 2000: 
        the optional columns in self.mapping (or authors.csv)
        """
        if options == 'mesh':
            # let's processing mesh terms
            # aggregate the results at the author level 
            meshdict = defaultdict(str)
            authors = set()
            for index, row in self.df_thin.iterrows():
                keys = str(row['authors']).rsplit(';')
                authors.update(keys)
                for key in keys:
                    meshdict[key] += str(row['mesh_terms']).rstrip()
                    
            # mapping author keys to its edge index 
            new_meshdict = {}
            for k, v in meshdict.items():
                key_int = self.mapdict.get(k, "empty")
                if key_int is not 'empty':
                    new_meshdict[key_int] = v
            final = dict(sorted(new_meshdict.items()))    
            # do the corpus aggregation
            mesh = list(final.values())
            meshlist = [x.split(';') for x in mesh]
            meshlist2 = []
            for z in meshlist:
                temp = [x[x.find(':')+1:] for x in z]
                meshlist2.append(temp)
            # this is for converting a list of list to a big list
            meshlist3 =  [" ".join(x) for x in meshlist2]
            corpus = meshlist3 
            
        elif options =='pubs':
            # crawl the pubmed articles' title infos first using the orignal authorfile
            # then set a feature representation 
            pubs_dict = {}
            # print(self.file_year)
            for k, v in self.authorID2Pubs.items():
                temp = [self.file_year.get(str(p))['title'] if p != '999999' else '' for p in v]
                # print(temp)
                temp = ' '.join(temp)
                pubs_dict[k] = temp                   
            dict_df = pd.DataFrame({'id': list(pubs_dict.keys()), 'title': list(pubs_dict.values())})
            # matching by id 
            corpus_df = self.mapping.merge(dict_df, on='id', how = 'left')
            # print(corpus_df.head())
            corpus = corpus_df['title'].tolist()
        else:
            print ('author feature representation not implemented')
        
        vectorizer = TfidfVectorizer(max_features = 2000) #control number of features here
        feat_x = vectorizer.fit_transform(corpus)
        # in the order of author, id, label, mask feature 
        self.mapping= pd.concat([self.mapping, pd.DataFrame(feat_x.toarray())], axis=1) 
        self.mapping.to_csv(self.savepath + 'authors.csv', index = False)
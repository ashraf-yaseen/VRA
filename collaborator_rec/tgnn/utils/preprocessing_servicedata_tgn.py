import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def preprocess(data_name):
  # we have the additional mask information
  u_list, i_list, ts_list, label_list, mask_ls = [], [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      u = int(e[0])
      i = int(e[1])

      ts = float(e[2])
      label = float(e[3])  # int(e[3])
      mask = float(e[4])

      feat = np.array([float(x) for x in e[5:]])

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      mask_ls.append(mask)
      idx_list.append(idx)#edge index

      feat_l.append(feat)
        
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'mask': mask_ls,
                       'idx': idx_list}), np.array(feat_l)


def reindex(df, bipartite=False):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df


def run(data_name, rand_node_feat = True, bipartite=False, path = 'service/'):
  Path("data/" + path).mkdir(parents=True, exist_ok=True)
  PATH = './data/{}{}.csv'.format(path, data_name)
  OUT_DF = './data/{}ml_{}.csv'.format(path,data_name)
  OUT_FEAT = './data/{}ml_{}.npy'.format(path,data_name)
  OUT_NODE_FEAT = './data/{}ml_{}_node.npy'.format(path, data_name)

  df, feat = preprocess(PATH)
  new_df = reindex(df, bipartite)

  empty = np.zeros(feat.shape[1])[np.newaxis, :]
  feat = np.vstack([empty, feat])

  max_idx = max(new_df.u.max(), new_df.i.max())
  if rand_node_feat:
      rand_feat = np.zeros((max_idx + 1, 172)) #why 172
      np.save(OUT_NODE_FEAT, rand_feat)  
        
  else:
    # do nothing, we've already processed in prepare_servicedata.py.
    pass
  

  new_df.to_csv(OUT_DF)
  np.save(OUT_FEAT, feat)
  # np.save(OUT_NODE_FEAT, rand_feat)

"""
In the mathematical field of graph theory, a bipartite graph (or bigraph) is a graph whose vertices can be divided into two disjoint and independent sets {\displaystyle U}U and {\displaystyle V}V such that every edge connects a vertex in {\displaystyle U}U to one in {\displaystyle V}V.
"""

import logging
import time
import math
from pathlib import Path
import sys
import os
import pickle
import random
import json

import numpy as np
import torch

# let's see which imports work
from .data_processing import get_data, compute_time_statistics
sys.path.insert(0, os.path.abspath('..'))
from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN

torch.manual_seed(2021)
np.random.seed(2021)


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class MLP(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:
            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, uniform, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))

  return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times


        
def make_log(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path('./data/' + args.savepath + "log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler('./data/' + args.savepath +'log/{}.log'.format(str(time.time())))
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


### service stage:modify from here
def compute_edge_probabilities(tgn, source_nodes, destination_nodes, negative_nodes, edge_times,
                                 edge_idxs, n_neighbors=20):
    """
    Compute probabilities for edges between sources and destination and between sources and
    negatives by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Probabilities for both the positive and negative edges
    """
    n_samples = len(source_nodes)
   
    # need to define compute_temporal_embeddings & afffinity score
    source_node_embedding, destination_node_embedding, _ = tgn.compute_temporal_embeddings(
      source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

    score = tgn.affinity_score(source_node_embedding,
                               destination_node_embedding).squeeze(dim=0)

    return score.sigmoid()


def recommend(logger, test_dst, probs,  \
                device, author_dict, firstk = 30, path = 'service/hulin_wu/', \
                f_name = 'hulin', l_name = 'wu', m_name = ''):

    # save results
    pred = torch.stack(probs).cpu().numpy().reshape(-1) # take the second columns
    idx = np.argsort(-pred)[:firstk]
    author_ids = np.take(test_dst,idx)
    
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
    
    with open('data/' + path + name + '_newresult.json', 'w') as fp:
        json.dump(result, fp)
    
    logger.info('produced {} recommended collaborators for {} '.format(firstk, name))    
    logging.shutdown()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()







######################### let's move all model related calculations to here
#### currently still under testing
"""
def compute_temporal_embeddings(source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs,\
                                n_nodes,\
                                embedding_module, n_layers,
                                mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst,\
                                n_neighbors=20, use_memory = False, memory_update_at_start = False, device = 'cpu', dyrep = False):
    '''
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    mean_time_shift_src, std_time_shift_src, are the numpy()?
    :return: Temporal embeddings for sources, destinations and negatives
    '''
 
    n_samples = len(source_nodes)
    # bunch of numpys
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes]) 
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times, edge_times])

    memory = None
    time_diffs = None
    if use_memory:
      if memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches
        memory, last_update = get_updated_memory(list(range(n_nodes)),
                                                      memory.messages)

      else:
        memory = memory.get_memory(list(range(n_nodes)))
        last_update = memory.last_update

      ### Compute differences between the time the memory of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      source_time_diffs = torch.LongTensor(edge_times) - last_update[
        source_nodes].long()
      source_time_diffs = (source_time_diffs - mean_time_shift_src) / std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times) - last_update[
        destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - mean_time_shift_dst) / std_time_shift_dst
      negative_time_diffs = torch.LongTensor(edge_times) - last_update[
        negative_nodes].long()
      negative_time_diffs = (negative_time_diffs - mean_time_shift_dst) / std_time_shift_dst

      time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                             dim=0).to(device)

    # Compute the embeddings using the embedding module
    node_embedding = embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs)
    # actual GPUSs 
    source_node_embedding = node_embedding[:n_samples].to(device) #modified by Ginny
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples].to(device)
    negative_node_embedding = node_embedding[2 * n_samples:].to(device)

    if use_memory:
      if memory_update_at_start:
        # Persist the updates to the memory only for sources and destinations (since now we have
        # new messages for them)
        update_memory(positives, memory.messages)

        assert torch.allclose(memory[positives], memory.get_memory(positives), atol=1e-5), \
          "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        memory.clear_messages(positives)

      unique_sources, source_id_to_messages = get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
      unique_destinations, destination_id_to_messages = get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
      if memory_update_at_start:
         memory.store_raw_messages(unique_sources, source_id_to_messages) #torch.cpu()
         memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else:
        update_memory(unique_sources, source_id_to_messages)
        update_memory(unique_destinations, destination_id_to_messages)

      if dyrep:
        source_node_embedding = memory[source_nodes].to(device)
        destination_node_embedding = memory[destination_nodes].to(device)
        negative_node_embedding = memory[negative_nodes].to(device)

    return source_node_embedding, destination_node_embedding, negative_node_embedding

  def compute_edge_probabilities(source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, \
                                 n_neighbors=20,\
                                 n_nodes, embedding_module, n_layers,\
                                 mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst,\
                                 use_memory = False, memory_update_at_start = False, device = 'cpu', dyrep = False, \
                                 ):
    '''
    Compute probabilities for edges between sources and destination and between sources and
    negatives by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Probabilities for both the positive and negative edges
    '''
    n_samples = len(source_nodes)
    source_node_embedding, destination_node_embedding, negative_node_embedding = compute_temporal_embeddings(
      source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors,\
                                n_nodes, embedding_module, n_layers,\
                                mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst,\
                                use_memory, memory_update_at_start, device , dyrep)
    # GPUs
    score = affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                torch.cat([destination_node_embedding,
                                           negative_node_embedding])).squeeze(dim=0)
    pos_score = score[:n_samples]
    neg_score = score[n_samples:]

    return pos_score.sigmoid(), neg_score.sigmoid()

  def update_memory(self, nodes, messages, message_aggregator, message_function):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = message_function.compute_message(unique_messages)

    # Update the memory with the aggregated messages
    self.memory_updater.update_memory(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps)

  def get_updated_memory( nodes, messages, message_aggregator, message_function, memory_updater):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = message_function.compute_message(unique_messages)

    updated_memory, updated_last_update = memory_updater.get_updated_memory(unique_nodes,
                                                                                 unique_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_memory, updated_last_update

  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features[edge_idxs].to(self.device) #modified by ginny

    source_memory = self.memory.get_memory(source_nodes) if not \
      self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory.get_memory(destination_nodes) if \
      not self.use_destination_embedding_in_message else destination_node_embedding

    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding],
                               dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
"""

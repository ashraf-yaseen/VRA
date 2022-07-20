""" contains utility functions needed for MC dropout, ensemble as well calibration
for tgn part
author: [Ginny](Jie.zhu@uth.tmc.edu)
date: 01/05/2022

"""
import os
import sys
import random
import time
import math
import datetime
import logging 
import pickle 
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.special import softmax
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.utils import check_consistent_length, column_or_1d
import seaborn as sns


import torch

# local, use from part0_grantRec_UQ, please import as below
#from .calibration_library import Metrics as calib
#from .calibration_library import visualization as visual
#from. calibration_library import recalibration as recalib
# directly use, please import below 
import calibration_library.Metrics as calib
#import calibration_library.visualization as visual
#import calibration_library.recalibration as recalib


# universal 
def sigmoid(x):
    sig = 1.0 / (1.0 + np.exp(-x))
    return sig


def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    
def flat_accuracy(preds, labels):
    #pred_flat = np.argmax(preds, axis=1).
    pred_flat = [1 if ele > 0.5 else 0 for ele in preds.flatten()]
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def perform_clf(sigmoid_np, labels_np, logger):
    #p = softmax(logits_np, axis =1)
    #prob1 = p[:,1]
    acc = flat_accuracy(sigmoid_np, labels_np)
    ap = average_precision_score(labels_np, sigmoid_np)
    auc = roc_auc_score(labels_np, sigmoid_np)
    ece = compute_calibration_error(y_true = labels_np, y_prob = sigmoid_np)
    print('Accuracy = {}, AUC = {}, AP = {} and ECE = {}'.format( acc, auc, ap, ece))
    logger.error('Accuracy = {}, AUC = {}, AP = {} and ECE = {}'.format(acc, auc, ap, ece))

    
## some references: 
# https://github.com/ethen8181/machine-learning/blob/master/model_selection/prob_calibration/calibration_module/utils.py
def create_binned_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Bin ``y_true`` and ``y_prob`` by distribution of the data.
    i.e. each bin will contain approximately an equal number of
    data points. Bins are sorted based on ascending order of ``y_prob``.
    Parameters
    ----------
    y_true : 1d ndarray
        Binary true targets.
    y_prob : 1d ndarray
        Raw probability/score of the positive class.
    n_bins : int, default 15
        A bigger bin number requires more data.
    Returns
    -------
    binned_y_true/binned_y_prob : 1d ndarray
        Each element in the list stores the data for that bin.
    """
    sorted_indices = np.argsort(y_prob)
    sorted_y_true = y_true[sorted_indices]
    sorted_y_prob = y_prob[sorted_indices]
    binned_y_true = np.array_split(sorted_y_true, n_bins)
    binned_y_prob = np.array_split(sorted_y_prob, n_bins)
    return binned_y_true, binned_y_prob

def get_bin_boundaries(binned_y_prob: List[np.ndarray]) -> np.ndarray:
    """
    Given ``binned_y_prob`` from ``create_binned_data`` get the
    boundaries for each bin.
    Parameters
    ----------
    binned_y_prob : list
        Each element in the list stores the data for that bin.
    Returns
    -------
    bins : 1d ndarray
        Boundaries for each bin.
    """
    bins = []
    for i in range(len(binned_y_prob) - 1):
        last_prob = binned_y_prob[i][-1]
        next_first_prob = binned_y_prob[i + 1][0]
        bins.append((last_prob + next_first_prob) / 2.0)

    bins.append(1.0)
    return np.array(bins)

def compute_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int=15,
    round_digits: int=4) -> float:
    """
    Computes the calibration error for binary classification via binning
    data points into the specified number of bins. Samples with similar
    ``y_prob`` will be grouped into the same bin. The bin boundary is
    determined by having similar number of samples within each bin.
    Parameters
    ----------
    y_true : 1d ndarray
        Binary true targets.
    y_prob : 1d ndarray
        Raw probability/score of the positive class.
    n_bins : int, default 15
        A bigger bin number requires more data. In general,
        the larger the bin size, the closer the calibration error
        will be to the true calibration error.
    round_digits : int, default 4
        Round the calibration error metric.
    Returns
    -------
    calibration_error : float
        RMSE between the average positive label and predicted probability
        within each bin.
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)

    binned_y_true, binned_y_prob = create_binned_data(y_true, y_prob, n_bins)

    # looping shouldn't be a source of bottleneck as n_bins should be a small number.
    bin_errors = 0.0
    for bin_y_true, bin_y_prob in zip(binned_y_true, binned_y_prob):
        avg_y_true = np.mean(bin_y_true)
        avg_y_score = np.mean(bin_y_prob)
        bin_error = (avg_y_score - avg_y_true) ** 2
        bin_errors += bin_error

    calibration_error = math.sqrt(bin_errors / n_bins)
    return round(calibration_error, round_digits)
    
    
#### part 1. MC dropout
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            


def train_batch_uq(batch, data, memory, gnn, link_pred, neighbor_loader, device, optimizer, criterion,\
                   min_dst_idx, max_dst_idx, assoc):
    

    optimizer.zero_grad()
    src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

    neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0), ),
                            dtype=torch.long, device=device)

    n_id = torch.cat([src, pos_dst, neg_dst]).unique()
    n_id, edge_index, e_id = neighbor_loader(n_id)
    assoc[n_id] = torch.arange(n_id.size(0), device=device)

    z, last_update = memory(n_id)
    z = gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])

    pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
    neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])
    loss = criterion(pos_out, torch.ones_like(pos_out))
    loss += criterion(neg_out, torch.zeros_like(neg_out))

    memory.update_state(src, pos_dst, t, msg)
    neighbor_loader.insert(src, pos_dst)
   
    loss.backward()
    optimizer.step()
    memory.detach()
    # total_loss = float(loss) * batch.num_events
    
    return loss, memory, gnn, link_pred, neighbor_loader
        

def train_epoch(epochs, memory, gnn, link_pred, neighbor_loader, \
                train_data, inference_data,  data, \
                min_dst_idx, max_dst_idx, assoc,\
                device, criterion, optimizer,\
                logger, i, load_path,\
                option = 'mcdrop', batch_size = 200):
    
    set_seed(i)
    training_stats = []
    total_t0 = time.time()
    best_auc = 0.

    
    for epoch in range(1, epochs + 1):
        
        # ========================================
        #               Training
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch, epochs))
        print('Training...')
        memory.train()
        gnn.train()
        link_pred.train()

        memory.reset_state()  # Start with a fresh memory.
        neighbor_loader.reset_state()  # Start with an empty graph.
        
        total_loss = 0.
        t0  = time.time()
        
        for batch in train_data.seq_batches(batch_size=batch_size):
            loss, memory, gnn, link_pred, neighor_loader = train_batch_uq(batch = batch, data = data, \
                                                                           memory = memory, gnn = gnn, link_pred = link_pred, \
                                                                           neighbor_loader = neighbor_loader, \
                                                                           device = device, optimizer = optimizer, \
                                                                           criterion = criterion, min_dst_idx = min_dst_idx, \
                                                                           max_dst_idx = max_dst_idx, assoc = assoc)
            total_loss += loss.item()
        total_loss = total_loss / train_data.num_events
        training_time = format_time(time.time() - t0)
        logger.error("  Training epcoh took: {}".format(training_time))
        logger.error('  Epoch: {}, training Loss: {}'.format(epoch, total_loss))
        
        # call evaluation
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on validation
        print("")
        print("Running Validation...")
        t0 = time.time()

        # Tracking variables 
        total_eval_loss = 0
        best_eval_loss = float('inf')
        
        memory.eval()
        gnn.eval()
        link_pred.eval()
        if option == 'mcdrop':
            enable_dropout(link_pred)
        
        predictions, true_labels = [], []

        # Predict 
        with torch.no_grad():
            for batch in inference_data.seq_batches(batch_size = batch_size):
                #depends on what outputs are returned in the model, unpack the values
                 memory, gnn, link_pred, \
                 neighbor_loader, \
                 y_true, output_logit, eval_loss, output_std = evaluate_batch_uq(batch = batch, \
                                                                 data = data, min_dst_idx = min_dst_idx, max_dst_idx = max_dst_idx,\
                                                                 assoc = assoc,\
                                                                 memory = memory, \
                                                                 gnn = gnn, \
                                                                 link_pred = link_pred, \
                                                                 neighbor_loader = neighbor_loader, \
                                                                 criterion = criterion, \
                                                                 device = device, \
                                                                 T = 1)
                 total_eval_loss += eval_loss

        total_eval_loss = total_eval_loss /inference_data.num_events
        validation_time = format_time(time.time() - t0)
        logger.error("  Validation epoch took: {}".format(validation_time))
        logger.error('  Epoch: {}, valid Loss: {}'.format(epoch, total_eval_loss))
        
    
        if total_eval_loss < best_eval_loss:
            best_eval_loss = total_eval_loss
            model_dict = {'memory': memory.state_dict(),
                          'gnn': gnn.state_dict(), 
                          'link_pred': link_pred.state_dict()}
            save_model_uq(model_path = load_path, model_dict = model_dict, logger = logger)

        # Record all statistics from this epoch.
        training_stats.append(
            {
            'epoch': epoch ,
            'Training Loss': total_loss,
            'Training Time': training_time,
            'Valid. Loss': total_eval_loss,
            'Validation Time': validation_time
             
            }
        )

        print("")
        print("Training complete!")
        print("Total training took {} (h:mm:ss)".format(format_time(time.time()-total_t0)))
        """
        checkpt = torch.load(load_path +'dict.pt')
        memory.load_state_dict(checkpt['memory'])
        gnn.load_state_dict(checkpt['gnn'])
        link_pred.load_state_dict(checkpt['link_pred'])
        """
        
    return  memory, gnn, link_pred,  neighbor_loader, training_stats


def plot_train(training_stats, save_path = ''):
    
    # Display floats with two decimal places.
    pd.set_option('precision', 2)
    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)
    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')
    # A hack to force the column headers to wrap.
    #df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])
    # Display the table.
    print(df_stats)
    df_stats.to_csv(save_path + 'df_stats.csv')
    
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')
    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)
    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(np.arange(100, step =10), rotation = 20)
    plt.savefig(save_path + 'train_val_loss.png')
    plt.show();

def save_model_uq(model_path, model_dict, logger):
        # saving models
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        logger.error("Saving models to %s" % model_path)
        #save model
        torch.save(model_dict,model_path+'dict.pt')

        
def evaluate_batch_uq(batch, data, min_dst_idx, max_dst_idx, assoc,\
                      memory, gnn, link_pred, neighbor_loader, criterion, device, T =1):

    src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

    neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0), ),
                                dtype=torch.long, device=device)

    n_id = torch.cat([src, pos_dst, neg_dst]).unique()
    n_id, edge_index, e_id = neighbor_loader(n_id)
    assoc[n_id] = torch.arange(n_id.size(0), device=device)
    z, last_update = memory(n_id)

    output_list = []
    eval_loss = 0.

    for i in range(T):

        z = gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])

        y_pred = torch.cat([pos_out, neg_out], dim=0).cpu()#.sigmoid().cpu()
        # print( 'y_pred dimeonsion:', y_pred.shape)
        output_list.append(y_pred)
        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))
        eval_loss += float(loss) #* batch.num_events
  
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)
        #print('y_true dimension:', y_true.shape)
    
    # check this part if correct
    output_logit = torch.stack(output_list, 0).mean(dim=0)
    output_std = torch.stack(output_list, 0).std(dim=0).numpy()
    eval_loss /= T
    

    memory.update_state(src, pos_dst, t, msg)
    neighbor_loader.insert(src, pos_dst)


    return memory, gnn, link_pred, neighbor_loader, y_true, output_logit, eval_loss, output_std   

  
            
def predictions_uq(memory, gnn, link_pred, inference_data, data, min_dst_idx, max_dst_idx, assoc, neighbor_loader, \
                   criterion, device, path, logger, batch_size = 200, option = 'mcdrop', T = 1):
    
    memory.eval()
    gnn.eval()
    link_pred.eval()
    if option == 'mcdrop':
        enable_dropout(link_pred)
        
    predictions, true_labels = [], []
    stds = []

    # Predict 
    with torch.no_grad():
        for batch in inference_data.seq_batches(batch_size = batch_size):
            #depends on what outputs are returned in the model, unpack the values
             memory, gnn, link_pred, \
             neighbor_loader, \
             y_true, output_logit, eval_loss, output_std = evaluate_batch_uq(batch = batch, \
                                                             data = data, min_dst_idx= min_dst_idx, max_dst_idx = max_dst_idx,\
                                                             assoc = assoc,\
                                                             memory = memory, \
                                                             gnn = gnn, \
                                                             link_pred = link_pred, \
                                                             neighbor_loader = neighbor_loader, criterion = criterion, \
                                                             device = device, \
                                                             T = T)

             # Store predictions and true labels
             predictions.append(output_logit.numpy())
             true_labels.append(y_true.unsqueeze(dim= -1).cpu().numpy())
             # let's see 
             stds.append(output_std)

    print('...DONE.')
    combine_predictions = np.concatenate(predictions, axis=0)
    # turn into probs 
    combine_predictions_probs = sigmoid(combine_predictions)
    #print(combine_predictions.shape)
    combine_true_labels = np.concatenate(true_labels, axis=0)
    #print(combine_true_labels.shape)
    # let's see 
    stds = np.concatenate(stds, axis = 0)                      
    #print(stds.shape)
    
    # Calculate accuracy
    # AUC and AP here too
    acc = flat_accuracy(combine_predictions_probs, combine_true_labels) 
    ap = average_precision_score(combine_true_labels,combine_predictions_probs)
    auc = roc_auc_score(combine_true_labels, combine_predictions_probs)
    ece = compute_calibration_error(combine_true_labels, combine_predictions_probs)
    print(acc, auc, ap, ece)
    logger.error('Test Accuracy: %.3f' % acc) 
    logger.error('Test AUC: %.3f' % auc)
    logger.error('Test AP: %.3f' % ap)
    logger.error('Test ECELoss: %.3f' % ece)
    
    np.save(path + 'combine_predictions', combine_predictions)
    np.save(path + 'combine_true_labels', combine_true_labels)
    # let's see
    np.save(path + 'std', stds) 
    
    return combine_predictions, combine_true_labels        
        

#### part 3. graphs
def combine_calibration_plot(names, y_test, y_probs, markers, fstyles, bins = 20, \
                             title_name = 'bert_based', additional_title = '', \
                             save_path = 'res_bert/' ):
    """
    inputs:
    1. names: a list of model names to refer to the framework experimented, e.g. ['No UQ', 'MC Dropout','Ensemble']
    2. y_test: self-explanatory
    3. y_probs: a list of predicted probability results of dimension (n x 2), corresponding to results produced by models in names
    4. # of bin: to bin the probability 
    for single comparision of no uq, MC dropout and ensemble, consider using:
    markers = ['s', 's', 's'], 
    fstyles = ['full','full','full']
    for before and after UQ adjusted, consider use:
    markers = ['s', 's', 's','o', "o"]
    fstyles = ['full','full','full', 'none','none']
    additional_title = 'before (◼) and after(○) UQ adjusted'
    """
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(5, 2)
    colors = plt.cm.get_cmap("Dark2")

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    for i,  name in enumerate(names):
        display = CalibrationDisplay.from_predictions(
            y_test,
            y_probs[i][:, -1],
            n_bins= bins,
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
            marker = markers[i],
            fillstyle = fstyles[i]
        )
        calibration_displays[name] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots for {} recommender {}".format(title_name, additional_title), fontsize=20)

    # Add histogram
    grid_positions = [(2, 0), (3, 0), (3, 1), (4, 0), (4, 1)]
    for i, name in enumerate(names):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])
        ax.hist(
            y_probs[i][:, -1],
            range=(0, 1),
            bins=bins,
            label=name,
            color=colors(i),
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    plt.savefig(save_path + title_name + '_calibration_test'+ additional_title + '.png', dpi = 300)
    plt.show();
    
    

def recalculate_score(y_logits, y_ci, option = 'mutliply', alpha1= 0.8, alpha2= 0.2):
    """
    mini function to re-calculate recommendation scores based on UQ
    multiply: using  y_ci = 1.96* std/sqrt(n)*2 for exp(-ci) * y_logits 
    weighted_avg: using alph1*y_logits + alpha2* 1./y_ci 
    """
    if option == 'multiply':
        new_logits = y_logits*np.exp(-y_ci)
    else:
        # scaler = MinMaxScaler(feature_range=(0.0001, 1), )
        # new_y_ci =  scaler.fit_transform(y_ci)
        new_logits = alpha1* y_logits + alpha2* 1./y_ci
        # new_logits[new_logits > 1.] = 1.
    
    return new_logits


#### when both methods are experimented:
def plot_all(args, num =5):
    """
    a mini function to plot out 5 results, based on a chosen UQ adjusted method
    """
    names = ['No UQ','MC Dropout','Ensemble','MC Dropout, UQ adjusted','Ensemble, UQ adjusted']

    # labels and predicted probs
    y_test = np.load(args.load_path + args.data_path +  'mcdrop0/combine_true_labels.npy')
    sigmoid_bsl = np.load( args.load_path + args.data_path + 'regular0/combine_predictions.npy') #previous experimentation path
    sigmoid_mcd = np.load(args.load_path + args.data_path + 'mcdrop0/combine_predictions.npy')
    sigmoid_ensem = np.load(args.load_path + args.data_path + 'ensemblecombine_predictions.npy')
    y_probs = [sigmoid_bsl, sigmoid_mcd,  sigmoid_ensem, sigmoid_mcd, sigmoid_ensem]

    mcd_ci = np.load(args.load_path + args.data_path +  'mcdrop0/std.npy')/np.sqrt(100) * 1.96 *2.0
    ens_ci = np.load(args.load_path + args.data_path+ 'ensemblestd.npy')/np.sqrt(5) * 1.96 *2.0
    cis = [0.0001, 0.0001, 0.0001, mcd_ci, ens_ci]
    alphas1 = [1, 1, 1, args.alpha1, args.alpha1]
    alphas2 = [0, 0, 0, args.alpha2, args.alpha2]
    
    y_probs_new = []
    scaler = MinMaxScaler() 
    for i, v in enumerate(y_probs):
        if i< 3:
            ci_s =cis[i]
        else:
            ci_s = scaler.fit_transform(cis[i])
        p = recalculate_score(v, ci_s, option = args.uqAdjusted, alpha1= alphas1[i], alpha2= alphas2[i])
        p = sigmoid(p)
        y_probs_new.append(p)
    
    if num == 3:
        add_title = 'calibration plot for no UQ, and two UQ'
    else: 
        add_title = 'before (◼) and after(○) UQ adjusted' + args.uqAdjusted
        
    combine_calibration_plot(names = names[:num], y_test = y_test, y_probs = y_probs_new[:num],
                           markers = ['s', 's', 's', 'o', 'o'][:num], fstyles = ['full','full','full','none','none'][:num],\
                           bins = 20, \
                           title_name = 'TGN_based, ', 
                           additional_title = add_title, \
                           save_path = args.load_path + args.data_path)

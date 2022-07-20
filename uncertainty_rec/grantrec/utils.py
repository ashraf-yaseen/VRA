""" contains utility functions needed for MC dropout, ensemble as well calibration
author: [Ginny](Jie.zhu@uth.tmc.edu)
date: 01/05/2022

"""
import torch
# import torch.nn.functional as F
import math
import sys
import random 
import logging 
import numpy as np
import pickle 
from typing import Dict, List, Tuple, Optional

from scipy.special import softmax
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_consistent_length, column_or_1d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

# local, use from part0_grantRec_UQ, please import as below
#from .calibration_library import Metrics as calib
#from .calibration_library import visualization as visual
#from. calibration_library import recalibration as recalib
# directly use, please import below 
import calibration_library.Metrics as calib
import calibration_library.visualization as visual
import calibration_library.recalibration as recalib

import sys
# parent sibling folder 
sys.path.append('../')
# grant-BERT related
from part0_GrantRec.eval_metrics import Metrics
    

#### part 1. MC dropout
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

            
            
#### part 2. ensemble 





#### part 3. graphs
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def perform_clf(logits_np, labels_np, load_path, logger):
    p = softmax(logits_np, axis =1)
    prob1 = p[:,1]
    acc = flat_accuracy(logits, labels_np)
    ap = average_precision_score(labels_np, prob1)
    auc = roc_auc_score(labels_np, prob1)
    ece = compute_calibration_error(y_true = labels_np, y_prob = prob1)
    print('Accuracy = {}, AUC = {}, AP = {} and ECE = {}'.format( acc, auc, ap, ece))
    logger.error('Accuracy = {}, AUC = {}, AP = {} and ECE = {}'.format(acc, auc, ap, ece))


def perform_rec(dp, similarity_dict, logger):

    mrr = Metrics(dp.citation).calculate_mrr(similarity_dict)
    print('MRR = {}'.format(mrr))

    r1 = Metrics(dp.citation).calculate_recall_at_k(similarity_dict, 1)
    r5 = Metrics(dp.citation).calculate_recall_at_k(similarity_dict, 5)
    print('recall@1 = {}, recall@5 = {}'.format(r1, r5))

    p1 = Metrics(dp.citation).calculate_precision_at_k(similarity_dict, 1)
    p5 = Metrics(dp.citation).calculate_precision_at_k(similarity_dict, 5)
    print('precision@1 = {}, precision@5 = {}'.format(p1, p5))

    map_ = Metrics(dp.citation).calculate_MAP_at_k(similarity_dict)
    print('MAP = {}'.format(map_))
    logger.error("MRR = {}, recall@1 = {}, recall@5 = {}, precision@1 = {}, precision@5 = {}, and MAP = {}".format(\
                                                                                      mrr, r1, r5, p1, p5, map_))
          
              
              
def calib_res(logits_np, labels_np, load_path, logger):
    
    ece_criterion = calib.ECELoss()
    ece = ece_criterion.loss(logits_np,labels_np, 15)
    print('ECE: %f' % (ece))

    softmaxes = softmax(logits_np, axis=1)
    ece_p = ece_criterion.loss(softmaxes,labels_np,15,False)
    print('ECE with probabilties %f' % (ece_p))

    mce_criterion = calib.MCELoss()
    mce = mce_criterion.loss(logits_np,labels_np)
    print('MCE: %f' % (mce))

    oe_criterion = calib.OELoss()
    oe = oe_criterion.loss(logits_np,labels_np)
    print('OE: %f' % (oe))

    sce_criterion = calib.SCELoss()
    sce = sce_criterion.loss(logits_np,labels_np, 15)
    print('SCE: %f' % (sce))

    ace_criterion = calib.ACELoss()
    ace = ace_criterion.loss(logits_np,labels_np,15)
    print('ACE: %f' % (ace))

    tace_criterion = calib.TACELoss()
    threshold = 0.01
    tace = tace_criterion.loss(logits_np,labels_np, threshold,15)
    print('TACE (threshold = %f): %f' % (threshold, tace))
    
    res_names = ['ECE', 'ECE with probabilties', 'MCE', 'OE', 'SCE', 'ACE', 'TACE(threshold = %f)'%(threshold)] 
    res_ls = [ece, ece_p, mce, oe, sce, ace, tace]
    res_dict = dict(zip(res_names, res_ls))
    with open(load_path + 'calibration.pickle', 'wb') as f:
        pickle.dump(res_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
       
    logger.error("ECE = {} , ECE with probabilties = {}, MCE = {}, OE = {}, SCE = {}, ACE = {}, TACE(threshold {}) = {}".format(\
                ece, ece_p, mce, oe, sce, ace, threshold, tace))
    logger.error('save reliability and confidence plots')
    
    # plots
    conf_hist = visual.ConfidenceHistogram()
    plt_test = conf_hist.plot(logits_np,labels_np,title="Confidence Histogram")
    plt_test.savefig(load_path + 'conf_histogram_test.png',bbox_inches='tight')
    plt_test.show();

    rel_diagram = visual.ReliabilityDiagram()
    plt_test_2 = rel_diagram.plot(logits_np,labels_np,title="Reliability Diagram")
    plt_test_2.savefig(load_path + 'rel_diagram_test.png',bbox_inches='tight')
    plt_test_2.show();


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
    
    
    
    
    
    
def temperature_scale(model, use_cuda, valid_loader, logits_np_test):
    scaled = recalib.ModelWithTemperature(model, use_cuda = use_cuda)
    scaled.set_temperature(valid_loader = valid_loader)
    scaled_logits = scaled.temperature_scale(torch.from_numpy(logits_np_test).to(use_cuda))# here logits should be test_logits tensor
    scaled_logits_np = scaled_logits.detach().cpu().numpy()
    return scaled_logits_np


def entropy_mi(dropout_predictions):
    """
    dropout predicitons should have the shape: T (use expand_dim to add dim =0)* n_samples * n_classes
    """
    # Calculating mean across multiple MCD forward passes 
    mean = np.mean(dropout_predictions, axis=0) # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes 
    variance = np.var(dropout_predictions, axis=0) # shape (n_samples, n_classes)

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes 
    entropy = -np.sum(mean*np.log(mean + epsilon), axis=-1) # shape (n_samples,)

    # Calculating mutual information across multiple MCD forward passes 
    mutual_info = entropy - np.mean(np.sum(-dropout_predictions*np.log(dropout_predictions + epsilon),
                                            axis=-1), axis=0) # shape (n_samples,)
    return variance, entropy, mutual_info



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
        #scaler = MinMaxScaler(feature_range=(0.0001, 1), )
        #new_y_ci =  scaler.fit_transform(y_ci)
        new_logits = alpha1* y_logits + alpha2* 1./y_ci
    return new_logits



#### when both methods are experimented:
def plot_all(args, num =5):
    """
    a mini function to plot out 5 results, based on a chosen UQ adjusted method
    """
    names = ['No UQ','MC Dropout','Ensemble','MC Dropout, UQ adjusted','Ensemble, UQ adjusted']

    # labels and predicted probs
    y_test = np.load(args.load_path +  'mcdrop0/combine_true_labels.npy')
    sigmoid_bsl = np.load('../part0_GrantRec/model_uq/combine_predictions.npy') #previous experimentation path
    sigmoid_mcd = np.load(args.load_path + 'mcdrop0/combine_predictions.npy')
    sigmoid_ensem = np.load(args.load_path + 'ensemble_combine_predictions.npy')
    y_probs = [sigmoid_bsl, sigmoid_mcd,  sigmoid_ensem, sigmoid_mcd, sigmoid_ensem]

    mcd_ci = np.load(args.load_path + 'mcdrop0/std.npy')
    ens_ci = np.load(args.load_path + 'ensemble_std.npy')
    cis = [0.0001, 0.0001, 0.0001, mcd_ci, ens_ci]
    alphas1 = [1, 1, 1, args.alpha1, args.alpha1]
    alphas2 = [0, 0, 0, args.alpha2, args.alpha2]
    
    y_probs_new = []
    for i, v in enumerate(y_probs):
        new_logits = recalculate_score(v, cis[i], option = args.uqAdjusted, alpha1= alphas1[i], alpha2= alphas2[i])
        p = softmax(new_logits, axis =1)[:, 1]
        y_probs_new.append(p)
    
    if num == 3:
        add_title = 'calibration plot for no UQ, and two UQ'
    else: 
        add_title = 'before (◼) and after(○) UQ adjusted' + args.uqAdjusted
        
    combine_calibration_plot(names = names[:num], y_test = y_test, y_probs = y_probs_new[:num],
                           markers = ['s', 's', 's', 'o', 'o'][:num], fstyles = ['full','full','full','none','none'][:num],\
                           bins = 20, \
                           title_name = 'BERT_based, ', 
                           additional_title = add_title, \
                           save_path = args.load_path + args.data_path)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 21:28:21 2022
@author:ginnyzhu
utility functions 

"""
#general
import logging
import pickle
import json
import dill
import os
import sys 
import pandas as pd
import numpy as np
import time
import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import requests
import re
import datefinder


#fancy libraries
import torch 
import torch.autograd as ag
from termcolor import colored
"""
need to fix master and year_weight stuff
"""
# parent sibling folder 
sys.path.append('../../')
from RFOrecsys.weight.year_weight import YearWeight
from functools import reduce
import itertools

##### Data loading related

#processing all the publications first 
def get_corpus_and_dict(df, id_col, filepickle, field1, field2, out_addr = '', name1 ='corpus', name2 ='corpus_dict'):
    '''
    get a list of id order
    do the (repetitive) content list appending
    then also ids to content dictionary
    '''
    id_ls = df[id_col].tolist()
    corpus = [] #check if a list or list of listst
    corpus_dict = {}
    for id in id_ls:
        temp = filepickle[str(id)][field1] + ' ' + filepickle[str(id)][field2]
        corpus.append(temp)
        corpus_dict[id] = temp
        
    print('length of the corpus', len(corpus))# 106,446
    print('sample of the corpus', corpus[:2])
    #dump these lists:
    with open(out_addr + name1+ '.pickle', 'wb') as f:
        pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_addr + name2+ '.pickle', 'wb') as f:
        pickle.dump(corpus_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return corpus, corpus_dict

#for the nih rfa processing 
def get_corpus_and_dict2(df, id_col, filecsv, file_id_col, field1, field2, out_addr = '', name1 ='corpus', name2 ='corpus_dict'):
    '''
    get a list of id order
    do the (repetitive) content list appending
    then also ids to content dictionary
    '''
    id_ls = df[id_col].tolist()
    corpus = [] #check if a list or list of listst
    corpus_dict = {}
 
    for id in id_ls:
        temp = filecsv.loc[filecsv[file_id_col]==id, field1].iloc[0] +' '+ filecsv.loc[filecsv[file_id_col]==id, field2].iloc[0]
        #break
        corpus.append(temp)
        corpus_dict[id] = temp
        
    print('length of the corpus', len(corpus))# 106,446
    print('sample of the corpus', corpus[:2])
    #dump these lists:
    with open(out_addr + name1+ '.pickle', 'wb') as f:
        pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_addr + name2+ '.pickle', 'wb') as f:
        pickle.dump(corpus_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return corpus, corpus_dict



######training related
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
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


def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    
def train_batch(Batch, model, optimizer, scheduler, cuda_device):# hyper_optim, vg): #optimizer, scheduler embScheduler,
    #modify furthere
    # emb related and actual model
    
    # Unpack this training batch from our dataloader. 
    # `batch` contains three pytorch tensors:
    #   [0]: input ids 
    #   [1]: attention masks
    #   [2]:type_ids 
    #   [3]:labels
    #   [4]:tfidf weights
    b_input_ids = Batch[0].cuda(cuda_device)
    b_input_mask = Batch[1].cuda(cuda_device)
    b_input_type_ids = Batch[2].cuda(cuda_device)
    b_labels = Batch[3].cuda(cuda_device)

    model.zero_grad() #because of the paramter tuning?

    loss, logits, Hidden = model(b_input_ids, 
                         token_type_ids=b_input_type_ids, 
                         attention_mask=b_input_mask, 
                         labels=b_labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
   

    return loss, logits




def evaluate_batch(Batch, model, cuda_device):
    
        b_input_ids = Batch[0].cuda(cuda_device)
        b_input_mask = Batch[1].cuda(cuda_device)
        b_input_type_ids = Batch[2].cuda(cuda_device)
        b_labels = Batch[3].cuda(cuda_device)
       
        #_, b_weights = model.test_tfidf(embBatch[0])
        #b_weights = torch.from_numpy(b_weights).cuda(1)
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass,
        with torch.no_grad():        

            (loss, logits, Hidden) = model(b_input_ids, 
                                   token_type_ids=b_input_type_ids, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
            #loss, logits  = model(embHidden, b_labels, b_weights)
            

        # Move logits and labels to CPU
        logits_cpu = logits.detach().cpu().numpy()
        label_ids_cpu = b_labels.cpu().numpy()

        return loss, logits_cpu, label_ids_cpu
    
    
    
    
    
def predictions(model, test_loader, use_cuda, path):
    model.eval()
    #model.bertEmb.eval()
    predictions , true_labels = [], []

    # Predict 
    for batch in test_loader:
        #depends on what outputs are returned in the model, unpack the values
        loss, logits_cpu, label_ids_cpu = evaluate_batch(Batch = batch, 
                                                         #embModel = model.bertEmb,  
                                                         model = model, 
                                                         cuda_device =use_cuda)

        # Store predictions and true labels
        predictions.append(logits_cpu)
        true_labels.append(label_ids_cpu)
        
    print('...DONE.')
    combine_predictions = np.concatenate(predictions, axis=0)
    combine_true_labels = np.concatenate(true_labels, axis=0)

    # Calculate accuracy
    acc = flat_accuracy(combine_predictions, combine_true_labels) #not great 0.930, just bert and different split
    print('Test Accuracy: %.3f' % acc) 
    
    np.save(path + 'combine_predictions', combine_predictions)
    np.save(path + 'combine_true_labels', combine_true_labels)
    
    return combine_predictions, combine_true_labels




    
def train(epochs, model, train_loader, valid_loader, optimizer, scheduler, use_cuda, tokenizer, args):
    
    training_stats = []
    #### Measure the total training time for the whole run.
    total_t0 = time.time()
    best_acc = 0.
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_train_loss = 0
        model.train()
        #model.bertEmb.train()

        # For each batch of training data...
        for step, batch in enumerate(train_loader):#change this later 

            # Progress update every 200 batches.
            if step % 200 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

            #b_weights = tfidf_vecs_weights[step* rfa.batch_size: (step+1)*rfa.batch_size]
            #depends on what outputs are returned in the model, unpack the values
            loss, logits = train_batch(Batch = batch, 
                                       model = model, 
                                       optimizer = optimizer,
                                       scheduler = scheduler, 
                                       cuda_device = use_cuda)

            total_train_loss += loss.item()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_loader)            
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))


        # call evaluation
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on validation
        print("")
        print("Running Validation...")
        t0 = time.time()
        # Put the model in evaluation mode--the dropout layers behave differently during evaluation 
        model.eval()
        #model.bertEmb.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0


        for batch in valid_loader: 
            #depends on what outputs are returned in the model, unpack the values
            loss, logits_cpu, label_ids_cpu = evaluate_batch(Batch = batch, 
                                                             #embModel = model.bertEmb,  
                                                             model = model, 
                                                             cuda_device =use_cuda)


            total_eval_loss += loss.item()
            total_eval_accuracy += flat_accuracy(logits_cpu, label_ids_cpu)

        avg_val_accuracy = total_eval_accuracy / len(valid_loader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(valid_loader)

        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        if avg_val_accuracy > best_acc:
            best_acc = avg_val_accuracy 
            save_model(model_path = args.load_path, model = model, tokenizer = tokenizer, args = args)

        # Record all statistics from this epoch.
        training_stats.append(
            {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
            }
        )

        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
        best_dict = torch.load(load_path + 'model.st')
        model.load_state_dict(best_dict)
        
    return training_stats, model
        

def save_model(model_path, model, tokenizer, args):
        # saving models
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        print("Saving model to %s" % model_path)
        
        #save model
        dill.dump(model, open(model_path +'model', 'wb'))
        torch.save(model.state_dict(), model_path + 'model.st')      
        model = model.module if hasattr(model, 'module') else model # distributed/parallel training
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(model_path, 'training_args.bin'))

            
def train_uq(epochs, model, train_loader, valid_loader, optimizer, scheduler, use_cuda, i, load_path, \
             tokenizer, logger, option = 'mcdrop'):
    
    set_seed(i)
    
    training_stats = []
    #### Measure the total training time for the whole run.
    total_t0 = time.time()
    best_acc = 0.
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_train_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_loader):#change this later 

            # Progress update every 200 batches.
            if step % 200 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

            #b_weights = tfidf_vecs_weights[step* rfa.batch_size: (step+1)*rfa.batch_size]
            #depends on what outputs are returned in the model, unpack the values
            loss, logits = train_batch(Batch = batch, 
                                       model = model, 
                                       optimizer = optimizer,
                                       scheduler = scheduler, 
                                       cuda_device = use_cuda)
            total_train_loss += loss.item()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_loader)            
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))



        # call evaluation
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on validation
        print("")
        print("Running Validation...")
        t0 = time.time()
        # Put the model in evaluation mode--the dropout layers behave differently during evaluation 
        model.eval()
        if option == 'mcdrop':
            enable_dropout(model)

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0



        for batch in valid_loader: 
            #depends on what outputs are returned in the model, unpack the values
            loss, logits_cpu, label_ids_cpu = evaluate_batch(Batch = batch, 
                                                             #embModel = model.bertEmb,  
                                                             model = model, 
                                                             cuda_device =use_cuda)


            total_eval_loss += loss.item()
            total_eval_accuracy += flat_accuracy(logits_cpu, label_ids_cpu)

        avg_val_accuracy = total_eval_accuracy / len(valid_loader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(valid_loader)

        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        if avg_val_accuracy > best_acc:
            best_acc = avg_val_accuracy 
            save_model_uq(model_path = load_path, model = model, tokenizer = tokenizer, logger = logger)

        # Record all statistics from this epoch.
        training_stats.append(
            {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
            }
        )

        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
        logger.error("epoch {}, training loss = {}, valid loss = {}, valid_accuracy = {}, train time = {}, valid time = {}".format(\
                     epoch_i +1,  avg_train_loss, avg_val_loss, avg_val_accuracy, training_time, validation_time))
        
        best_dict = torch.load(load_path + 'pytorch_model.bin')
        model.load_state_dict(best_dict)
        
    return training_stats, model
        
        
 
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
       
              
        
        
def save_model_uq(model_path, model, tokenizer, logger):
        # saving models
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        print("Saving model to %s" % model_path) 
        logger.error("Saving model and tokenizer to %s" % model_path)
        
        #save model
        dill.dump(model, open(model_path + 'model', 'wb'))
        # torch.save(model.state_dict(), model_path + 'model.st')
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        # torch.save(args, os.path.join(model_path, 'training_args.bin'))




def evaluate_batch_uq(Batch, model, cuda_device, T =1):
    
        b_input_ids = Batch[0].cuda(cuda_device)
        b_input_mask = Batch[1].cuda(cuda_device)
        b_input_type_ids = Batch[2].cuda(cuda_device)
        b_labels = Batch[3].cuda(cuda_device)
       
        output_list = []
        loss_list = []
        with torch.no_grad():
            for i in range(T):
                # total of T forward passes 
                (loss, logits, Hidden) = model(b_input_ids, 
                                       token_type_ids=b_input_type_ids, 
                                       attention_mask=b_input_mask,
                                       labels=b_labels)
                output_list.append(torch.unsqueeze(logits, dim= 0))
                loss_list.append(torch.unsqueeze(loss, dim= 0))#loss)
                
            output_mean = torch.cat(output_list, 0).mean(dim=0)
            output_variance = torch.cat(output_list, 0).var(dim=0).mean().item()
            confidence = output_mean.data.cpu().numpy().max()
            output_loss = torch.cat(loss_list,0).mean()

        # Move logits and labels to CPU
        logits_cpu = output_mean.detach().cpu().numpy()
        label_ids_cpu = b_labels.to('cpu').numpy()

        return output_loss, logits_cpu, label_ids_cpu, output_variance, confidence            
            
def predictions_uq(model, test_loader, use_cuda, path, option = 'mcdrop', T = 1):
    
    model.eval()
    if option == 'mcdrop':
        enable_dropout(model)
        
    predictions, true_labels = [], []
    variances, confis = [], []

    # Predict 
    for batch in test_loader:
        #depends on what outputs are returned in the model, unpack the values
        loss, logits_cpu, label_ids_cpu, variance, confidence = evaluate_batch_uq(Batch = batch, 
                                                         model = model, 
                                                         cuda_device =use_cuda, T= T)

        # Store predictions and true labels
        predictions.append(logits_cpu)
        true_labels.append(label_ids_cpu)
        # let's see 
        variances.append(variance)
        confis.append(confidence)
                         
          
    print('...DONE.')
    combine_predictions = np.concatenate(predictions, axis=0)
    combine_true_labels = np.concatenate(true_labels, axis=0)
    # let's see 
    variances = np.asarray(variances)
    confis =  np.asarray(confis)
                               
                               

    # Calculate accuracy
    acc = flat_accuracy(combine_predictions, combine_true_labels) #not great 0.930, just bert and different split
    print('Test Accuracy: %.3f' % acc) 
    
    np.save(path + 'combine_predictions', combine_predictions)
    np.save(path + 'combine_true_labels', combine_true_labels)
    # let's see
    np.save(path + 'variances', variances) 
    np.save(path + 'confidences', confis)
    
    return combine_predictions, combine_true_labels, variances, confis 



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
    plt.xticks(list(df_stats.index))
    plt.savefig(save_path + 'train_val_loss.png')
    plt.show();

    
    
    

    
def prob(x):
    """Compute prob from logits."""
    return np.exp(x[:,-1]) / (np.exp(x[:,-1]) +1) 
    
def create_smilarity_dict(citation_df, combine_predictions, save_path):
  
    probas = prob(combine_predictions)
    ## comment below when actual testing
    # citation_df = citation_df.iloc[:len(probas), :].copy()
    citation_df['pred_prob1'] = probas
    pred_flat = np.argmax(combine_predictions, axis=1).flatten()
    citation_df['pred'] = pred_flat
    citation_df2  = citation_df[citation_df['pred'] == 1]
    
    pred_grouped = citation_df2.groupby("pmid").agg(**{
                          "rfas_recom": pd.NamedAgg(column='rfaid', aggfunc=lambda x:x.to_list()),
                          "rfas_prob": pd.NamedAgg(column='pred_prob1', aggfunc=lambda x:x.to_list())               
                          }).reset_index()
    print(pred_grouped.shape)
    print(pred_grouped.head())
    
    pred_grouped.to_csv(save_path + 'pred_grouped.csv', index = False)
    pred_grouped['leng'] = pred_grouped['rfas_prob'].str.len()
    max_length = pred_grouped['leng'].max()
     
    similarity_dict = {}
    for _,row in pred_grouped.iterrows():
        d = dict(zip(row['rfas_recom'], row['rfas_prob']))
        sort_d = dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
        #print(sort_d)
        #break
        similarity_dict[row['pmid']] = sort_d 
    with open(save_path + 'similarity_dict', 'wb') as handle:
        pickle.dump(similarity_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return similarity_dict, max_length 
    
    

       
def create_smilarity_dict2(citation_df, combine_predictions, save_path):
  
    #probas = prob(combine_predictions)
    probas = combine_predictions[:, -1]
    citation_df['pred_prob1'] = probas
    pred_flat = np.argmax(combine_predictions, axis=1).flatten()
    citation_df['pred'] = pred_flat
    citation_df2  = citation_df[citation_df['pred'] == 1]
    
    pred_grouped = citation_df2.groupby("pmid").agg(**{
                          "rfas_recom": pd.NamedAgg(column='rfaid', aggfunc=lambda x:x.to_list()),
                          "rfas_prob": pd.NamedAgg(column='pred_prob1', aggfunc=lambda x:x.to_list())               
                          }).reset_index()
    print(pred_grouped.shape)
    print(pred_grouped.head())
    
    pred_grouped.to_csv(save_path + 'pred_grouped.csv', index = False)
    pred_grouped['leng'] = pred_grouped['rfas_prob'].str.len()
    max_length = pred_grouped['leng'].max()
     
    similarity_dict = {}
    for _,row in pred_grouped.iterrows():
        d = dict(zip(row['rfas_recom'], row['rfas_prob']))
        sort_d = dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
        #print(sort_d)
        #break
        similarity_dict[row['pmid']] = sort_d 
    with open(save_path + 'similarity_dict', 'wb') as handle:
        pickle.dump(similarity_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return similarity_dict, max_length 
    
    
    
    
## need to modify something from here
"""
def processCVrfa(f_name, l_name, m_name, rfa_ls, processed_path = 'data/', output_path= 'results_v0/'):
    
    pubs_output = './resources/'+ l_name + '.pickle'
    cv_path= './resources/'+  l_name +'CV.pdf'
    output_path = output_path +l_name+'_output.json'
        
    old_time = datetime.now()
    final_data = Master().process(cv_path, pubs_output, f_name, l_name, m_name)
    final_data['researcher_name'] = f_name + ' ' + m_name + ' ' + l_name
    import json
    json.dump(final_data, open(output_path, 'w'), indent=4)
    print(datetime.now()-old_time)
    
    #read in what we are interested in
    with open(processed_path + f_name + l_name +  '_pubDetails', 'rb') as f:
        pubD = pickle.load(f)
    with open(processed_path + f_name + l_name + '_pmids', 'rb') as f:
        pmids = pickle.load(f) 
        
    #create pairs and save it 
    pairs = list(itertools.product(pmids, rfa_ls))
    df = pd.DataFrame(pairs, columns =['pmid','rfaid'])  
    df.to_csv(processed_path + f_name +l_name + '_df.csv', index = False)
    
    #create dataloader
    rfa = RFADataProcessForPred(path = 'data_processed/', whospubs= f_name+ l_name + '_pubDetails' ,
                 load_pretrained = False, load_path = '../DLrec/model_save_v6/', nosplit= True)

    rfa.dataframize_()
    #then to dataloader 
    test_loader, test_pr = rfa.dataloaderize_() #dataloader right 
    
    return rfa, test_loader, test_pr
"""

def evaluate_batch_woLabels(Batch, model, cuda_device, haslabels= False):
    
        b_input_ids = Batch[0].cuda(cuda_device)
        b_input_mask = Batch[1].cuda(cuda_device)
        b_input_type_ids = Batch[2].cuda(cuda_device)
        if haslabels: 
            b_labels = Batch[3].cuda(cuda_device)
            
        with torch.no_grad():
            if haslabels:
                (loss, logits, Hidden) = model(b_input_ids, 
                                       token_type_ids=b_input_type_ids, 
                                       attention_mask=b_input_mask,
                                       labels=b_labels)       
            
            else:
                (logits, Hidden) = model(b_input_ids, 
                                       token_type_ids=b_input_type_ids, 
                                       attention_mask=b_input_mask)
            

        # Move logits and labels to CPU
        logits_cpu = logits.detach().cpu().numpy()
        outputs = (logits_cpu,)
        if haslabels:
            label_ids_cpu = b_labels.to('cpu').numpy()
            outputs = (logits_cpu, label_ids_cpu, loss)

        return outputs


def getPredRes(model, test_loader, use_cuda, f_name, l_name):
    
    outpath = f_name.lower() + l_name.lower() + '/'
    
    model.eval()
    predictions , true_labels = [], []

    # Predict 
    for batch in test_loader:
        #depends on what outputs are returned in the model, unpack the values
        logits_cpu, = evaluate_batch_woLabels(Batch = batch, 
                                    model = model, 
                                    cuda_device =use_cuda)

        # Store predictions and true labels
        predictions.append(logits_cpu)

    print('...DONE.')
    np.save(outpath + 'predictions', predictions)
    #stack all batches
    combine_predictions = np.concatenate(predictions, axis=0)
    #get the probablities
    probas = prob(combine_predictions)
    #get the predictions labels 
    pred_flat = np.argmax(combine_predictions, axis=1).flatten()
    np.save(outpath + 'probas', probas)
    np.save(outpath +  'pred_flat', pred_flat)
    
    #save results toghether with the paired list
    df = pd.read_csv(outpath + 'df.csv')
    dfnew = df.iloc[:len(probas),:].copy()
    dfnew['probas'] = probas
    dfnew['pred_flat'] = pred_flat
    dfnew.to_csv(outpath+  'preds_df.csv', index = False )
    
    return pred_flat, probas


def getGrantTypes(linktopage):
    """
    linktopage: should be a single, syntaxtly correct url 
    return: a list of content, could be len==1 or more
    """
    page = requests.get(linktopage.strip(" ;")) # make sure url is clean
    try:
        soup = BeautifulSoup(page.content, 'html.parser')
        conts = []
        for elem in soup(href=lambda href: href and "Search_Type=Activity" in href):
            content = elem.parent.a.text + ' ' +elem.parent.a.find_next_sibling(text=True).strip(' ,')
            conts.append(content)
        return conts
    except:
        print('connection problem')
        
        
def mergeRes(pathTilName):
    """
    mini function to combine recommendation results with  clustering rsults 
    """
    with open( pathTilName + '_clusteredRes.json', 'r') as f:
        file1 = json.load(f)
    recommended_ls = []
    for k,v in file1.items():
        if len(v) != 0:
            recommended_ls.append(v)
    with open( pathTilName + '_outwname.json', 'r') as f:
        file2 = json.load(f)
    # file 2 is the target
    temp = file2['data']
    new_data = []
    assert len(temp) == len(recommended_ls)
    for i, each in enumerate(temp):
        #print(each)
        each.update({'recommended_rfas': recommended_ls[i]})
        new_data.append(each)
    file2['data'] = new_data
    # write it out 
    with open(pathTilName+  '_output.json', 'w') as f:
        json.dump(file2, f, indent =4) 
        

category_types = {'R': 'Research Grants (R series)',
                 'K': 'Career Development Awards (K series)', 
                 'T': 'Research Training and Fellowships (T & F series)',
                 'F': 'Research Training and Fellowships (T & F series)',
                 'P': 'Program Project/Center Grants (P series)'}
        
        
def renameRes(file):
    """
    mini function to rewrite recommendation results with clustering rsults _output.json
    """
    with open( file, 'r') as f:
        file1 = json.load(f)
    # file 2 is the target
    temp = file1['data']
    new_data = []
    for i, each in enumerate(temp):
        #print(each)
        each_rfa = each.pop('recommended_rfas')# this dictionary 
        # renaming a bunch of things
        new_each_rfa  = []
        for j, ele in enumerate(each_rfa):
            ele['id'] = ele.pop('rfa_id')
            ele['link'] = ele.pop('rfa_links')
            ele['title'] = ele.pop('rfa_title')
            ele['purpose'] = ele.pop('rfa_purpose')
            ztemp = ele.pop("rfa_releaseDate")
            matches = list(datefinder.find_dates(ztemp))
            ele['release_date'] = matches[0].strftime('%m/%d/%Y') if len(matches)> 0 else ztemp
            ztemp = ele.pop("rfa_ExpireDate")
            matches = list(datefinder.find_dates(ztemp))
            ele['expired_date'] = matches[0].strftime('%m/%d/%Y') if len(matches)> 0 else ztemp          
            candis = re.findall(r'[a-zA-Z]{2}[0-9]{1}|[a-zA-Z]{1}[0-9]{2}',ele['title'])
            if len(candis) == 0:
                candi = 'UNK'
            elif len(candis) ==1:
                candi = candis[0]
            else:
                candi = '/'.join(candis)
            ele['Activity_Code'] = candi
            ele['Organization'] = ''
            ele['Clinical_Trials'] = ''
            ele['matched_words'] = ''
            ele['rank'] = j
            ele['score'] = 999
            ele["agency"]= "NIH"
            ele['category'] = ele['Activity_Code']
            ele['type'] = category_types.get(ele['Activity_Code'][0], 'Other')
            ele.pop('rfa_types')
            new_each_rfa.append(ele)         
        each.update({'recommended_rfos': new_each_rfa}) #list of 20
        new_data.append(each)
    file1['data'] = new_data
    # write it out
    print('writing to {}'.format(file))
    with open(file, 'w') as f:
        json.dump(file1, f, indent =4)        
        
        
        
        
        
        
        
        
        
        
        
        
        
def clustered_recom(f_name, m_name, l_name, data_path, logger, top = 20):
    
    # publication cluster information
    out_path = f_name.lower() + l_name.lower() + '/'
    clusters = pickle.load(open( out_path +'clusteredPubs', 'rb'))
    all_pmids = pickle.load(open( out_path+'pmids', 'rb'))
    all_years = pickle.load(open( out_path+ 'pubYrs', 'rb'))
    #get the rfas 
    rfas = pd.read_csv(data_path + 'processed_nih_grants_only.csv') 
    
    #year weight formula
    yearW = YearWeight()
    
    #get the predictions with 1s only
    pred_df = pd.read_csv(out_path + 'preds_df.csv')
    pred_df_keep = pred_df.loc[pred_df['pred_flat'] == 1,:].copy()
    pred_df_keep.reset_index(drop= True, inplace = True)
    
    
    clusters_rec = {}
    for i in clusters: # enumerate key
        cluster_i = {}
        idx = clusters[i]
        temp_pmid = np.take(all_pmids, idx)
        temp_yr = np.take(all_years, idx)
        
        if len(idx)==1:
            #in this case, we only have 1 publication per cluster, no need to consider the year effect
            recom_df = pred_df_keep.loc[pred_df_keep['pmid']==temp_pmid[0],:].copy() 
            recom_df.reset_index(drop = True, inplace = True)
            results = recom_df['probas'].sort_values(ascending= False)
            
            
        else:
            #in this case, yes, we need to consider the year effect
            #do separatetly for each pmid, merge and sort
            recom_dfs = []
            for j, pmid in enumerate(temp_pmid):
                df = pred_df_keep.loc[pred_df_keep['pmid']== int(pmid),:].copy()                 
                df.reset_index(drop = True, inplace = True)
                df['score'] = yearW.calculate_weight(vec = df.probas, publication_year= int(temp_yr[j]))
                #print(df.head())
                recom_dfs.append(df[['rfaid','score']])
            recom_df = reduce(lambda x, y: pd.merge(x, y, on = 'rfaid'), recom_dfs) 
            recom_df = pd.DataFrame(recom_df.T.groupby([s.split('_')[0] for s in recom_df.T.index.values]).sum().T, 
                                   columns = ['rfaid', 'score'])
            #print(recom_df.head())
            #print([s.split('_')[0] for s in recom_df.T.index.values])
            results = recom_df['score'].sort_values(ascending= False) 
            #print(results)
                                 
        recom_rfas = np.take(recom_df['rfaid'].tolist(), list(results.index))
        #print(recom_rfas[:10])
        cluster_i['rfa_id'] = recom_rfas[:top]  
        #get the titles 
        recom_rfa_titles = [rfas.query('funding_opportunity_number==@rfa')['funding_opportunity_title'].values[0]\
                            for rfa in recom_rfas[:top]]
        cluster_i['rfa_title'] = recom_rfa_titles[:top]
        #get the links
        recom_rfa_links = [rfas.query('funding_opportunity_number==@rfa')['link_to_additional_information'].values[0].strip('; ') \
                           for rfa in recom_rfas[:top]]
        cluster_i['rfa_links'] = recom_rfa_links
        recom_rfa_types = [getGrantTypes(link) for link in recom_rfa_links] # a list of lists
        cluster_i['rfa_types'] = recom_rfa_types
        # cluster_i['rfa_link'] = recom_rfa_links[:top]                         
        # get the grant types, this needs information of the 'link_to_additional_information' and crawl under 'Activity Codes'
        'view-source:https://grants.nih.gov/grants/guide/rfa-files/RFA-DK-19-501.html'
        # get the purpose
        recom_rfa_desc = [rfas.query('funding_opportunity_number==@rfa')['description'].values[0] for rfa in recom_rfas[:top]]
        cluster_i['rfa_purpose'] = recom_rfa_desc#[:top]
        # get the dates  
        recom_rfa_releases = [rfas.query('funding_opportunity_number==@rfa')['posted_date'].values[0].strip() for \
                              rfa in recom_rfas[:top]]
        cluster_i['rfa_releaseDate'] = recom_rfa_releases#[:top]
        recom_rfa_exps = [rfas.query('funding_opportunity_number==@rfa')['current_closing_date_for_applications'].values[0].strip() \
                          for rfa in recom_rfas[:top]]
        cluster_i['rfa_ExpireDate'] = recom_rfa_exps#[:top]
        # repeat the column name as dictionary key, so each cluster has a list of same-structured dictionary as results 
        clusters_rec['cluster'+str(i)] = list(pd.DataFrame.from_dict(cluster_i).T.to_dict().values())
    # write the results
    if m_name.strip() =='':
        name = f_name + '_' + l_name 
    else:
        name = f_name + '_' + m_name + '_' + l_name  
    logger.error('{}: total {} clusters, recommended {} rfas per cluster'.format(name, len(clusters), top))    
    logging.shutdown()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()      
    with open(out_path + name + '_clusteredRes.json', 'w') as f:
            json.dump(clusters_rec, f)
            
    # combine results 
    mergeRes(pathTilName = out_path + name)
    renameRes(file = out_path + name)
                                 
    return clusters_rec
    




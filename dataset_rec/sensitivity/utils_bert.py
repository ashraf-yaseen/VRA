#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 21:28:21 2019
@author:ginnyzhu
utility functions 

"""
#general
import pickle
import dill
import os
import pandas as pd
import numpy as np
import time
import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

#fancy libraries
import torch 
import torch.autograd as ag
from termcolor import colored




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

#for the geodata processing 
def get_corpus_and_dict4(df, id_col, filepickle, out_addr = '', textfield1 ='title', textfield2= 'abstract',
                         name1 ='corpus', name2 ='corpus_idls'):
    '''
    get a list of id order
    do the (repetitive) content list appending
    then also ids to content dictionary
    '''
    id_ls = df[id_col].tolist()
    corpus = [] #check if a list or list of listst
    corpus_ls = id_ls
 
    for id in id_ls:
        #if str(id) in filepickle:
        temp_dict = filepickle[str(id)]
        temp = str(temp_dict[textfield1]) + ' ' + str(temp_dict[textfield2])
        corpus.append(temp)
        
    print('length of the corpus', len(corpus))# 
    print('sample of the corpus', corpus[:2])
    #dump these lists:
    with open(out_addr + name1+ '.pickle', 'wb') as f:
        pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_addr + name2+ '.pickle', 'wb') as f:
        pickle.dump(corpus_ls, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return corpus, corpus_ls





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

    
def train_batch(Batch, model, optimizer, scheduler, cuda_device):
    
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
    
    """
    first_grad = ag.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
    hyper_optim.compute_hg(model, first_grad)
    for params, gradients in zip(model.parameters(), first_grad):
         params.grad = gradients
    optimizer.step()
    hyper_optim.hyper_step(vg.val_grad(model))
    scheduler.step()
    clear_grad(model)
    
    """

    return loss, logits





def train(epochs, model, train_loader, valid_loader, optimizer, scheduler, use_cuda, args):
    
    training_stats = []
    #best_acc = 0.
    best_auc = 0.
    #### Measure the total training time for the whole run.
    total_t0 = time.time()
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
            if step % 2000 == 0 and not step == 0:
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

        predictions, true_labels = [], []
        
        for batch in valid_loader: 
            #depends on what outputs are returned in the model, unpack the values
            loss, logits_cpu, label_ids_cpu = evaluate_batch(Batch = batch, 
                                                             #embModel = model.bertEmb,  
                                                             model = model, 
                                                             cuda_device =use_cuda)


            total_eval_loss += loss.item()
            total_eval_accuracy += flat_accuracy(logits_cpu, label_ids_cpu)
            # Store predictions and true labels
            predictions.append(logits_cpu)
            true_labels.append(label_ids_cpu)
            

        avg_val_accuracy = total_eval_accuracy / len(valid_loader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(valid_loader)
        combine_predictions = np.concatenate(predictions, axis=0)
        combine_true_labels = np.concatenate(true_labels, axis=0)
        val_auc = roc_auc_score(combine_true_labels, combine_predictions[:,-1])
        print(" Auc:{0:.2f}".format(val_auc))

        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
        
        if val_auc > best_auc:
            best_auc = val_auc
            save_model(model_path = args.load_path, model = model, args = args)

        # Record all statistics from this epoch.
        training_stats.append(
            {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Valid. Auc': val_auc,
            'Training Time': training_time,
            'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    return training_stats
        

def save_model(model_path, model, args):
        # saving models
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        print("Saving model to %s" % model_path)     
        
        #save model
        dill.dump(model, open(model_path +'model', 'wb'))
        torch.save(model.state_dict(), model_path + 'model.st')      
        model = model.module if hasattr(model, 'module') else model # distributed/parallel training
        model.save_pretrained(model_path)
        #model.tokenizer.save_pretrained(model_path)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(model_path, 'training_args.bin'))


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
    plt.show();

    
    
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
        label_ids_cpu = b_labels.to('cpu').numpy()

        return loss, logits_cpu, label_ids_cpu
    
    
def predictions(model, test_loader, use_cuda, path):
    
    t0 = time.time()
    
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
    auc = roc_auc_score(combine_true_labels, combine_predictions[:,-1])
    pred_time = format_time(time.time() - t0)
    print('Test Accuracy: %.3f' % acc) 
    print('Test Auc: %.3f' % auc)
    print("  Test/prediction took: {:}".format(pred_time))
    
    np.save(path + 'combine_predictions', combine_predictions)
    np.save(path + 'combine_true_labels', combine_true_labels)
    
    return combine_predictions, combine_true_labels


def prob(x):
    """Compute prob from logits."""
    return np.exp(x[:,-1]) / (np.exp(x[:,-1]) +1) 
      
def create_smilarity_dict(citation_df, combine_predictions, save_path):
    
    citation_df = citation_df.iloc[:combine_predictions.shape[0],:].copy()
    probas = prob(combine_predictions)
    citation_df['pred_prob1'] = probas
    pred_flat = np.argmax(combine_predictions, axis=1).flatten()
    #citation_df = citation_df.iloc[:]
    citation_df['pred'] = pred_flat
    citation_df2  = citation_df[citation_df['pred'] == 1].copy() #take predicted 1s only 
    
    pred_grouped = citation_df2.groupby("pmid").agg(**{
                          "data_recom": pd.NamedAgg(column='dataid', aggfunc=lambda x:x.to_list()),
                          "data_prob": pd.NamedAgg(column='pred_prob1', aggfunc=lambda x:x.to_list())               
                          }).reset_index()
    print(pred_grouped.shape)
    print(pred_grouped.head())
    
    pred_grouped.to_csv(save_path + 'pred_grouped.csv', index = False)
    pred_grouped['leng'] = pred_grouped['data_prob'].str.len()
    max_length = pred_grouped['leng'].max()
     
    similarity_dict = {}
    for _,row in pred_grouped.iterrows():
        d = dict(zip(row['data_recom'], row['data_prob']))
        sort_d = dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
        #print(sort_d)
        #break
        similarity_dict[str(row['pmid'])] = sort_d #string key to be consistent
    with open(save_path + 'similarity_dict', 'wb') as handle:
        pickle.dump(similarity_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return similarity_dict, max_length 

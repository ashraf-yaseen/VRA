#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 21:28:21 2019
@author:ginnyzhu
utility functions 

"""
#general
import pickle
import logging
import dill
import os
import pandas as pd
import numpy as np
import time
import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns


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
        best_dict = torch.load(args.load_path + 'model.st')
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
        # output_vars = []
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
            output_std = torch.cat(output_list, 0).std(dim=0).cpu().numpy()
            confidence = output_mean.data.cpu().numpy().max()
            output_loss = torch.cat(loss_list,0).mean()

        # Move logits and labels to CPU
        logits_cpu = output_mean.detach().cpu().numpy()
        label_ids_cpu = b_labels.to('cpu').numpy()
        

        return output_loss, logits_cpu, label_ids_cpu, output_std, confidence            
            
def predictions_uq(model, test_loader, use_cuda, path, option = 'mcdrop', T = 1):
    
    model.eval()
    if option == 'mcdrop':
        enable_dropout(model)
        
    predictions, true_labels = [], []
    stds, confis = [], []

    # Predict 
    for batch in test_loader:
        #depends on what outputs are returned in the model, unpack the values
        loss, logits_cpu, label_ids_cpu, std, confidence = evaluate_batch_uq(Batch = batch, 
                                                         model = model, 
                                                         cuda_device =use_cuda, T= T)

        # Store predictions and true labels
        predictions.append(logits_cpu)
        true_labels.append(label_ids_cpu)
        # let's see 
        stds.append(std)
        confis.append(confidence)
                         
          
    print('...DONE.')
    combine_predictions = np.concatenate(predictions, axis=0)
    combine_true_labels = np.concatenate(true_labels, axis=0)
    # let's see 
    stds = np.concatenate(stds, axis = 0)
    confis =  np.asarray(confis)
                               
                               

    # Calculate accuracy
    acc = flat_accuracy(combine_predictions, combine_true_labels) #not great 0.930, just bert and different split
    print('Test Accuracy: %.3f' % acc) 
    
    np.save(path + 'combine_predictions', combine_predictions)
    np.save(path + 'combine_true_labels', combine_true_labels)
    # let's see
    np.save(path + 'std', stds) 
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
    
    
    
    
    
    

    
    
    
    
    




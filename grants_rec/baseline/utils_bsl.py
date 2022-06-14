import pickle
import os
import sys
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import accuracy_score, roc_curve, auc
from gensim import corpora
from gensim.summarization import bm25
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# local 
from eval_metrics import Metrics





def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)

#### part0. data preparations
def load_data(data_path):
    rfas = pd.read_csv(data_path + 'processed_nih_grants_only.csv')
    pubs = pickle.load(open(data_path +'processed_pubs.pickle', 'rb'))
    mix_df = pd.read_csv(data_path+'pairs_mixed.csv')
    train_idx = pickle.load(open(data_path +'train_idx.ls', 'rb'))
    valid_idx = pickle.load(open(data_path +'valdi_idx.ls', 'rb'))
    test_idx = pickle.load(open(data_path +'test_idx.ls', 'rb'))
    #citation data
    train_citation = pickle.load(open(data_path + 'train_citation_data.dict', 'rb'))
    valid_citation = pickle.load(open(data_path + 'valid_citation_data.dict', 'rb'))
    citation = pickle.load(open(data_path + 'citation_data.dict', 'rb'))
    #final mix citation data
    train_mixed = pickle.load(open(data_path + 'train_mix_citation_data.dict', 'rb'))
    valid_mixed = pickle.load(open(data_path + 'valid_mix_citation_data.dict', 'rb'))
    test_mixed = pickle.load(open(data_path + 'test_mix_citation_data.dict', 'rb'))
    return (rfas, pubs, mix_df, \
            train_idx, valid_idx, test_idx, \
            train_citation, valid_citation, citation, \
            train_mixed, valid_mixed, test_mixed)



#### part modeling part data
def process_rfa_corpus(df, vectorizer, outpath, \
                       columns = ['processed_funding_opportunity_title', 'processed_description'],\
                       load_pretrained = False):
    "columns: has to be in order of title and then descrption " 
    if not load_pretrained:
        df['processed'] = df[columns].agg(' '.join, axis=1)
        rfa_corpus = df['processed'].tolist()
        rfa_tfidf = vectorizer.fit_transform(rfa_corpus) 
        pickle.dump(vectorizer, open(outpath + "vectorizer", "wb"))
        pickle.dump(rfa_tfidf, open(outpath + "rfa_tfidf.pickle", "wb"))
        rfa_ids = df['funding_opportunity_number'].tolist()
        pickle.dump(rfa_ids, open(outpath + "rfa_ids.ls", "wb"))
    else:
        rfa_tfidf = pickle.load(open(outpath +'rfa_tfidf.pickle', 'rb'))
        rfa_ids = pickle.load(open(outpath +'rfa_ids.ls', 'rb'))
        vectorizer = pickle.load(open(outpath +'vectorizer', 'rb'))
    return rfa_tfidf, rfa_ids, vectorizer

def process_pub_query(idx, mix_df, pubs, vectorizer, outpath, idx_name = 'valid_', fields= ['ptitle', 'pabstract'],\
                     load_pretrained = False):
    """
    idx: idx to records from mix_df 
    pubs: the pickle file containing detailed metas about publications
    """
    path = outpath + idx_name
    if not load_pretrained: 
        select = mix_df.iloc[idx].copy()
        pmids = select['pmid'].unique().tolist()
        queries = []
        for pmid in pmids: 
            text = pubs.get(str(pmid), "")
            if text == '':
                pmids.remove(pmid)
                continue
            else:
                temp = text[fields[0]]
                for i in range(1, len(fields)):
                    temp = temp + ' ' + text[fields[i]]       
                queries.append(temp)
        pubs_tfidf = vectorizer.transform(queries) 
        pickle.dump(pubs_tfidf, open(path + "pubs_tfidf.pickle", "wb"))
        pickle.dump(pmids, open(path + "pmids.ls", "wb"))
    else:
        pubs_tfidf = pickle.load(open(path + "pubs_tfidf.pickle", 'rb'))
        pmids = pickle.load(open(path  +'pmids.ls', 'rb'))
    return pubs_tfidf, pmids 


def make_citations(idx, mix_df, match_only = True, idx_name = 'valid_', outpath = 'newdata/'):
    """
    idx: idx to records from mix_df 
    return: either citation dictionary, or the mixed dictionary that bert is on 
    """
    select = mix_df.iloc[idx].copy()
    # pmids = select['pmid'].unique().tolist()
    if match_only:
        citation_df = select[select['match'] == 1].copy()
    else:
        citation_df = select
    grouped = citation_df.groupby("pmid").agg(**{
                          "rfas": pd.NamedAgg(column='rfaid', aggfunc=lambda x:x.to_list()),               
                          }).reset_index()
    grouped['leng'] = grouped['rfas'].str.len()
    max_length = grouped['leng'].max()
    print('max length of rfas: {}'.format(max_length))
    
    citation_dict = dict(zip(grouped['pmid'], grouped['rfas']))
    pickle.dump(citation_dict, open(outpath + idx_name+  "citation_data.dict", "wb"))

    return citation_dict


def process_rfa_corpus_bm25(df, outpath,\
                       columns = ['processed_funding_opportunity_title', 'processed_description'],\
                       load_pretrained = False):
    "columns: has to be in order of title and then descrption " 
    if not load_pretrained:
        df['processed'] = df[columns].agg(' '.join, axis=1)
        rfa_corpus = df['processed'].tolist()
        texts = [doc.split() for doc in rfa_corpus] # you can do preprocessing as removing stopwords
        dictionary = corpora.Dictionary(texts)
        pickle.dump(dictionary, open(outpath + "dictionary", "wb"))
        rfa_vecs = [dictionary.doc2bow(text) for text in texts]
        vectorizer = bm25.BM25(rfa_vecs)
        pickle.dump(vectorizer, open(outpath + "vectorizer", "wb"))
        pickle.dump(rfa_vecs, open(outpath + "rfa_bm25.pickle", "wb"))
        rfa_ids = df['funding_opportunity_number'].tolist()
        pickle.dump(rfa_ids, open(outpath + "rfa_ids.ls", "wb"))
    else:
        rfa_vecs = pickle.load(open(outpath +  "rfa_bm25.pickle", 'rb'))
        rfa_ids = pickle.load(open(outpath +'rfa_ids.ls', 'rb'))
        vectorizer = pickle.load(open(outpath +'vectorizer', 'rb'))
        dictionary = pickle.load(open(outpath +'dictionary', 'rb'))
    return rfa_vecs, rfa_ids, vectorizer, dictionary


def process_pub_query_bm25(idx, mix_df, pubs, vectorizer, dictionary, outpath, \
                      idx_name = 'valid_', fields= ['ptitle', 'pabstract'], load_pretrained = False):
    """
    idx: idx to records from mix_df 
    pubs: the pickle file containing detailed metas about publications
    """
    path = outpath + idx_name
    if not load_pretrained: 
        select = mix_df.iloc[idx].copy()
        pmids = select['pmid'].unique().tolist()
        queries = []
        for pmid in pmids: 
            text = pubs.get(str(pmid), "")
            if text == '':
                pmids.remove(pmid)
                continue
            else:
                temp = text[fields[0]]
                for i in range(1, len(fields)):
                    temp = temp + ' ' + text[fields[i]]       
                queries.append(temp)
        pubs_vecs = [dictionary.doc2bow(query.split()) for query in queries]
        scores = [vectorizer.get_scores(query_doc) for query_doc in pubs_vecs]
        pickle.dump(pubs_vecs, open(path + "pubs_bm25.pickle", "wb"))
        pickle.dump(scores, open(path + "scores.pickle", "wb"))
        pickle.dump(pmids, open(path + "pmids.ls", "wb"))
    else:
        pubs_vecs = pickle.load(open(path + "pubs_bm25.pickle", 'rb'))
        scores =  pickle.load(open(path + "scores.pickle", 'rb'))
        pmids = pickle.load(open(path  +'pmids.ls', 'rb'))
    return scores, pmids, pubs_vecs 


def process_rfa_corpus_d2v(df, outpath, args, \
                       columns = ['processed_funding_opportunity_title', 'processed_description'],\
                       load_pretrained = False):
    "columns: has to be in order of title and then descrption " 
    fname = outpath + "d2v"
    if not load_pretrained:
        df['processed'] = df[columns].agg(' '.join, axis=1)
        rfa_corpus = df['processed'].tolist()
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(rfa_corpus)]
        model = Doc2Vec(documents, \
                        vector_size=args.vector_size, min_count= args.min_count, \
                        workers= args.workers, epochs= args.epochs)
        model.save(fname)      
        rfa_ids = df['funding_opportunity_number'].tolist()
        pickle.dump(rfa_ids, open(outpath + "rfa_ids.ls", "wb"))
    else:
        model = Doc2Vec.load(fname)
        rfa_ids = pickle.load(open(outpath +'rfa_ids.ls', 'rb'))
    return model, rfa_ids

def process_pub_query_d2v(idx, mix_df, pubs, model, outpath, \
                      idx_name = 'test_', fields= ['ptitle', 'pabstract'], load_pretrained = False):
    """
    idx: idx to records from mix_df 
    pubs: the pickle file containing detailed metas about publications
    """
    path = outpath + idx_name
    if not load_pretrained: 
        select = mix_df.iloc[idx].copy()
        pmids = select['pmid'].unique().tolist()
        queries = []
        for pmid in pmids: 
            text = pubs.get(str(pmid), "")
            if text == '':
                pmids.remove(pmid)
                continue
            else:
                temp = text[fields[0]]
                for i in range(1, len(fields)):
                    temp = temp + ' ' + text[fields[i]]       
                queries.append(temp)
        inferred_vector = [model.infer_vector(query.split()) for query in queries]
        sims = [model.docvecs.most_similar([inf], topn= len(model.docvecs)) for inf in inferred_vector]
        pickle.dump(inferred_vector, open(path + "pubs_d2v.pickle", "wb"))
        pickle.dump(sims, open(path + "sims.pickle", "wb"))
        pickle.dump(pmids, open(path + "pmids.ls", "wb"))
    else:
        inferred_vector = pickle.load(open(path + "pubs_d2v.pickle", 'rb'))
        sims =  pickle.load(open(path + "sims.pickle", 'rb'))
        pmids = pickle.load(open(path  +'pmids.ls', 'rb'))
    return sims, pmids, inferred_vector




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


def select_data_nb(idx, mix_df, pubs, rfas, outpath, idx_name = 'train_', load_pretrained = False):
    """
    using idx and mix_df to select records,
    using pubs and rfas to create training corpus
    return processed x and y 
    """
    path = outpath + idx_name
    if not load_pretrained:
        pub_corpus, _= get_corpus_and_dict(df= mix_df,id_col = 'pmid',
                                             filepickle= pubs, field1= 'ptitle', field2 ='pabstract',
                                             out_addr = outpath, name1 ='pub_corpus', name2 ='pub_corpus_dict')
        rfa_corpus, _ = get_corpus_and_dict2(df= mix_df, id_col = 'rfaid',
                                               filecsv = rfas, file_id_col =  'funding_opportunity_number',
                                               field1= 'processed_funding_opportunity_title', 
                                               field2 ='processed_description',
                                               out_addr = outpath, name1 ='rfa_corpus', name2 ='rfa_corpus_dict')
        pubs_select = np.asarray(pub_corpus)[idx]
        rfas_select = np.asarray(rfa_corpus)[idx]
        X =  np.char.add(pubs_select, rfas_select)
        select = mix_df.iloc[idx].copy()
        pmids = select['pmid'].to_numpy()
        rfaids = select['rfaid'].to_numpy()
        y = select['match'].to_numpy()
        np.save(path + 'X.npy', X)
        np.save(path + 'y.npy', y)
        np.save(path + 'pmids.npy', pmids)
        np.save(path + 'rfaids.npy', rfaids)
    else:
        X = np.load(path + 'X.npy' ,allow_pickle=True) 
        y = np.load(path + 'y.npy', allow_pickle=True)
        pmids = np.load(path + 'pmids.npy', allow_pickle=True)
        rfaids = np.load(path + 'rfaids.npy', allow_pickle=True)
    return X, y, pmids, rfaids





####part2. train and evaluations 
def sim_recommend(corpus_vecs, corpus_ids, query_vecs, query_ids, mix_dict, outpath, \
                  mode = 'approx', query_name = 'valid_', top = 7):
    """ 
    corpus_vecs: rfa corpus 
    query_vecs: pubs vector
    top: number of rfas to recommend: 7 because max_length in testing/trainging is 7 
    mix_dict: the complete test dictionary, including both truth and false rfas (used in BERT)
    return: similarity_dict with pmid as keys, and rfa-dict ={id: score} as the values 
    """
    cos_scores = linear_kernel(query_vecs, corpus_vecs)
    # get top 20 per rows 
    similarity_dict = {}
    for i, pmid in enumerate(query_ids):
        temp = dict(zip(corpus_ids, cos_scores[i]))
        sort_d = dict(sorted(temp.items(), key=lambda x: x[1], reverse=True)[:top])
        testing_temp = mix_dict.get(pmid, 'notFound') 
        if testing_temp != 'notFound':
            # adding in testing pairs and try again 
            add_d = {key:temp[key] for key in testing_temp}
            if mode == 'approx':
                take = {k: sort_d[k] for k in list(sort_d)[:top-len(add_d)]}
                add_d.update(take)
            # else we only checked on bert tested values 
            sort_d = dict(sorted(add_d.items(), key=lambda x: x[1], reverse=True))
        similarity_dict[pmid] = sort_d
    pickle.dump(similarity_dict, open(outpath + query_name + mode + "similarity_dict", "wb"))   
    
    return similarity_dict



def sim_recommend_bm25(corpus_ids, query_scores, query_ids, mix_dict, outpath, \
                  mode = 'approx', query_name = 'test_', top = 10):
    """ 
    query_scores: pubs scores produced by process_pub_query_bm25
    top: number of rfas to recommend: 10 
    mix_dict: the complete test dictionary, including both truth and false rfas (used in BERT)
    return: similarity_dict with pmid as keys, and rfa-dict = {id: score} as the values 
    """
    # get top  per rows 
    similarity_dict = {}
    for i, pmid in enumerate(query_ids):
        temp = dict(zip(corpus_ids, query_scores[i]))
        sort_d = dict(sorted(temp.items(), key=lambda x: x[1], reverse=True)[:top])
        testing_temp = mix_dict.get(pmid, 'notFound') 
        if testing_temp != 'notFound':
            # adding in testing pairs and try again 
            add_d = {key:temp[key] for key in testing_temp}
            if mode == 'approx':
                take = {k: sort_d[k] for k in list(sort_d)[:top-len(add_d)]}
                add_d.update(take)
            # else we only checked on bert tested values 
            sort_d = dict(sorted(add_d.items(), key=lambda x: x[1], reverse=True))
        similarity_dict[pmid] = sort_d
    pickle.dump(similarity_dict, open(outpath + query_name + mode + "similarity_dict", "wb"))   
    
    return similarity_dict



def sim_recommend_d2v(corpus_ids, sims, query_ids, mix_dict, outpath, \
                  mode = 'approx', query_name = 'test_', top = 10):
    """ 
    sims: a list of tuple produced by process_pub_query_d2v, already sorted using scores
    top: number of rfas to recommend: 10 
    mix_dict: the complete test dictionary, including both truth and false rfas (used in BERT)
    return: similarity_dict with pmid as keys, and rfa-dict = {id: score} as the values 
    """
    # get top  per rows 
    similarity_dict = {}
    for i, pmid in enumerate(query_ids):
        ls_argmax = [x[0] for x in sims[i]]
        ls_scores = [x[1] for x in sims[i]]
        corpus_ids_sort = np.take(corpus_ids, ls_argmax)
        sort_d = dict(zip(corpus_ids_sort[:top], ls_scores[:top]))
        temp = dict(zip(corpus_ids_sort, ls_scores))
        testing_temp = mix_dict.get(pmid, 'notFound') 
        if testing_temp != 'notFound':
            # adding in testing pairs and try again 
            add_d = {key:temp[key] for key in testing_temp}
            if mode == 'approx':
                take = {k: sort_d[k] for k in list(sort_d)[:top-len(add_d)]}
                add_d.update(take)
            # else we only checked on bert tested values 
            sort_d = dict(sorted(add_d.items(), key=lambda x: x[1], reverse=True))
        similarity_dict[pmid] = sort_d
    pickle.dump(similarity_dict, open(outpath + query_name + mode + "similarity_dict", "wb"))   
    
    return similarity_dict


def create_smilarity_dict(pmids, rfaids, combine_predictions_probas, save_path, idx_name = 'train_'):
    
    path = save_path + idx_name
    probas = combine_predictions_probas[:, -1]
    # citation_df['pred_prob1'] = probas
    pred_flat = np.argmax(combine_predictions_probas, axis=1).flatten()
    # citation_df['pred'] = pred_flat
    
    df = pd.DataFrame(list(zip(pmids, rfaids, probas, pred_flat)),
                                              columns =['pmid','rfaid', 'pred_proba', 'pred'])
    
    df2  = df[df['pred'] == 1]
    
    pred_grouped = df2.groupby("pmid").agg(**{
                          "rfas_recom": pd.NamedAgg(column='rfaid', aggfunc=lambda x:x.to_list()),
                          "rfas_prob": pd.NamedAgg(column='pred_proba', aggfunc=lambda x:x.to_list())               
                          }).reset_index()
    print(pred_grouped.shape)
    print(pred_grouped.head())
    
    pred_grouped.to_csv(path + 'pred_grouped.csv', index = False)
    pred_grouped['leng'] = pred_grouped['rfas_prob'].str.len()
    max_length = pred_grouped['leng'].max()
    print(max_length)
     
    similarity_dict = {}
    for _,row in pred_grouped.iterrows():
        d = dict(zip(row['rfas_recom'], row['rfas_prob']))
        sort_d = dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
        similarity_dict[row['pmid']] = sort_d 
    with open(path + 'similarity_dict', 'wb') as handle:
        pickle.dump(similarity_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return similarity_dict, max_length 


def mini_auc(y, probas):
    fpr, tpr, thresholds = roc_curve(y, probas[:, 1])
    auc_ = auc(fpr, tpr)
    return auc_


def print_metrics(citation, similarity_dict, logger, ks = [1, 5]):
    
    """
    citation: actual citations
    similarity_dict:recommendations 
    logger: logger to record the results
    ks: a list of two ks for calculating metrics@k, default 1 and 5(max_length for recommendations) 
    """ 
    print('MRR:')
    mrr = Metrics(citation).calculate_mrr(similarity_dict)
    print(mrr)
    logger.error('MRR: {}'.format(mrr))

    print('recall@1, recall@5:')
    r1 = Metrics(citation).calculate_recall_at_k(similarity_dict, ks[0])
    r2 = Metrics(citation).calculate_recall_at_k(similarity_dict, ks[-1])
    print(r1)
    print(r2)
    logger.error('recall@{}: {}, recall@{}: {}'.format(ks[0], r1, ks[-1], r2))

    print('precision@1, precision@5:')
    p1 = Metrics(citation).calculate_precision_at_k(similarity_dict, ks[0])
    p2 = Metrics(citation).calculate_precision_at_k(similarity_dict, ks[-1])
    print(p1)        
    print(p2)
    logger.error('precision@{}: {}, precision@{}: {}'.format(ks[0], p1, ks[-1], p2))

    print('MAP:')
    map_ = Metrics(citation).calculate_MAP_at_k(similarity_dict)
    print(map_)
    logger.error('map: {}'.format(map_))

"""
#temporary functions: used only once during development 
def modify_json(path='evalService/', file1 = 'clusteredPubs', file2 = '_clusteredRes.json', f='Cici', m ='', l='Bauer' ):
    folder_name= f.lower() + l.lower() + '/'
    if m.strip() == '':
        file2_com = f + '_'+ l+ file2
    else:
        file2_com = f + '_'+m+"_"+ l+ file2
        
    clus = pickle.load(open(path + folder_name + file1, 'rb'))
    with open(path +folder_name + file2_com, "r") as read_file:
        clusres = json.load(read_file)
        
    new = {'cluster'+ str(list(clus.keys())[i]): v for i, (k, v) in enumerate(clusres.items())} 
    
    with open(path +folder_name + file2_com, "w") as jsonFile:
        json.dump(new, jsonFile)
        
def rename(path='evalService/', file1 = 'outwname.json', f='Cici', m ='', l='Bauer'):
    folder_name= f.lower() + l.lower() 
    import os
    old_file = os.path.join(path + folder_name, file1)
    if m.strip() == '':
        file1_new = f + '_'+ l+'_' + file1
    else:
        file1_new = f + '_'+m+"_"+ l+ '_' + file1
    
    new_file = os.path.join(path + folder_name, file1_new)
    os.rename(old_file, new_file)

"""

    
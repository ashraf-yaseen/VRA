# %%
import glob
import nltk

import data_cleaning as dataClean
import multiprocessing as mp
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi as bm

# %%
def setLemmatizer():
    lemmatizer = WordNetLemmatizer()

    return lemmatizer

def processCorpus(data):
    wiki_token = data.copy()
    wiki_token = wiki_token[["Conference Title", "WikiCFP Tags", "Conference Description"]]
    wiki_token.columns = ["title", "tags", "description"]
    wiki_token.fillna('', inplace = True)
    wiki_token["soup"] = wiki_token["tags"] + " " + wiki_token["description"]

    return wiki_token

def preprocess_sentences(text):
    text = text.lower()
    temp_sent = []
    words = nltk.word_tokenize(text)
    tags = nltk.pos_tag(words)
    for i, word in enumerate(words):
        if tags[i][1] in verb_codes:
            lemmatized = lemmatizer.lemmatize(word, 'v')
        else:
            lemmatized = lemmatizer.lemmatize(word)
        if lemmatized not in stop_words and lemmatized.isalpha():
            temp_sent.append(lemmatized)
            
    finalsent = ' '.join(temp_sent)
    finalsent = finalsent.replace("n't", " not")
    finalsent = finalsent.replace("'m", " am")
    finalsent = finalsent.replace("'s", " is")
    finalsent = finalsent.replace("'re", " are")
    finalsent = finalsent.replace("'ll", " will")
    finalsent = finalsent.replace("'ve", " have")
    finalsent = finalsent.replace("'d", " would")
    
    return finalsent

def multiprocessApply(function, corpus, data_col, new_col):
    with mp.Pool(mp.cpu_count()) as pool:
        corpus[new_col] = pool.map(function, corpus[data_col])

    return corpus

def createBMObject(corpus, data_col):
    wiki_bm = corpus[data_col].copy().to_list()
    wiki_bm_token = [doc.split(" ") for doc in wiki_bm]

    bm25 = bm(wiki_bm_token)

    return bm25

def getBM25Ranks(processing_func, query, model):
    query = processing_func(query)
    token_query = query.split(" ")
    doc_scores = model.get_scores(token_query)

    return doc_scores

def getRecs(model_scores, top_n, corpus):
    results = np.argsort(model_scores)[-top_n:]
    results = np.flip(results)

    final_recs = corpus.loc[corpus.index[results]]

    return final_recs

if __name__ == "__main__":
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("wordnet")
    nltk.download("stopwords")

    stop_words = set(stopwords.words("english"))

    verb_codes = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}

    lemmatizer = setLemmatizer()

    wikicfp = dataClean.read_folder("/workspaces/VRA/conference_rec/wikicfp_csv")
    wikicfp = dataClean.unique_confs_per_year(wikicfp)
    wikicfp = dataClean.better_dates(wikicfp)

    wiki_token = processCorpus(wikicfp)

    wiki_token = multiprocessApply(preprocess_sentences, wiki_token, "soup", "processed_soup")

    bm25_model = createBMObject(wiki_token, "processed_soup")

    query = "AMIA 2022 Informatics Summit: From discovering innovative methods to learning from exciting real-world applications, AMIA 2022 Informatics Summit attendees will experience the full range of cutting-edge work in translational informatics and clinical data science from inception to implementation. This conference is the ideal setting for researchers, educators, data scientists, software developers and analysts, students, and industry professionals. The size of the conference makes it ideal for developing meaningful new connections and partnerships while learning practical advice to solve real-world challenges. New to the AMIA 2022 Informatics Summit, we have expanded upon the previous Informatics Implementation track to include it as a new theme: Applied Informatics. In addition to selecting one of the three core Programmatic Tracks (Clinical Research Informatics, Data Science, Translational Bioinformatics), authors/presenters can also choose to designate their submission as part of the Applied Informatics theme to highlight the crucial application and implementation focus of their work. This is first time the Informatics Summit convenes outside of San Francisco. We are confident Chicago will bring new collaborations and connections. We look forward to receiving your submissions"

    query_scores = getBM25Ranks(preprocess_sentences, query, bm25_model)

    recs = getRecs(query_scores, 10, wikicfp)
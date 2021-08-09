import glob
import nltk

import data_cleaning as dataClean
import multiprocessing as mp
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi as bm

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

    lemmatizer = dataClean.setLemmatizer()

    wikicfp = dataClean.readFolder("/workspaces/VRA/conference_rec/wikicfp_csv")
    wikicfp = dataClean.uniqueConfsPerYear(wikicfp)
    wikicfp = dataClean.betterDates(wikicfp)

    wiki_token = dataClean.processCorpus(wikicfp)

    wiki_token = dataClean.multiprocessApply(dataClean.preprocess_sentences, wiki_token, "soup", "processed_soup")

    bm25_model = createBMObject(wiki_token, "processed_soup")

    query = "AMIA 2022 Informatics Summit: From discovering innovative methods to learning from exciting real-world applications, AMIA 2022 Informatics Summit attendees will experience the full range of cutting-edge work in translational informatics and clinical data science from inception to implementation. This conference is the ideal setting for researchers, educators, data scientists, software developers and analysts, students, and industry professionals. The size of the conference makes it ideal for developing meaningful new connections and partnerships while learning practical advice to solve real-world challenges. New to the AMIA 2022 Informatics Summit, we have expanded upon the previous Informatics Implementation track to include it as a new theme: Applied Informatics. In addition to selecting one of the three core Programmatic Tracks (Clinical Research Informatics, Data Science, Translational Bioinformatics), authors/presenters can also choose to designate their submission as part of the Applied Informatics theme to highlight the crucial application and implementation focus of their work. This is first time the Informatics Summit convenes outside of San Francisco. We are confident Chicago will bring new collaborations and connections. We look forward to receiving your submissions"

    query_scores = getBM25Ranks(dataClean.preprocess_sentences, query, bm25_model)

    recs = getRecs(query_scores, 10, wikicfp)
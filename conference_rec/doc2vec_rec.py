import gensim.models
import glob
import nltk

import data_cleaning as dataClean
import multiprocessing as mp
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def createDoc2VecObject(corpus, data_col):
    wiki_gen = corpus[data_col].copy().to_list()
    wiki_gen_token = [doc.split(" ") for doc in wiki_gen]

    wiki_gensim = [gensim.models.doc2vec.TaggedDocument(d, [i]) for i, d in enumerate(wiki_gen_token)]

    return wiki_gensim

def createModel(corpus):
    model = gensim.models.doc2vec.Doc2Vec(vector_size = 50, min_count = 2, epochs = 40)
    model.build_vocab(corpus)
    model.train(corpus, total_examples = model.corpus_count, epochs = model.epochs)

    return model

def getDoc2VecScores(processing_func, query, model):
    query = processing_func(query)
    token_query = query.split(" ")

    vector = model.infer_vector(token_query)
    sims = model.dv.most_similar([vector], topn=len(model.dv))

    return sims

def getDoc2VecRecs(model_scores, top_n, corpus):
    results = pd.DataFrame(model_scores[:top_n], columns = ["idx", "cos_sim"])
    final_recs = corpus.loc[corpus.index[results["idx"]]]

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

    d2v_corpus = createDoc2VecObject(wiki_token, "processed_soup")

    d2v_model = createModel(d2v_corpus)

    query = "AMIA 2022 Informatics Summit: From discovering innovative methods to learning from exciting real-world applications, AMIA 2022 Informatics Summit attendees will experience the full range of cutting-edge work in translational informatics and clinical data science from inception to implementation. This conference is the ideal setting for researchers, educators, data scientists, software developers and analysts, students, and industry professionals. The size of the conference makes it ideal for developing meaningful new connections and partnerships while learning practical advice to solve real-world challenges. New to the AMIA 2022 Informatics Summit, we have expanded upon the previous Informatics Implementation track to include it as a new theme: Applied Informatics. In addition to selecting one of the three core Programmatic Tracks (Clinical Research Informatics, Data Science, Translational Bioinformatics), authors/presenters can also choose to designate their submission as part of the Applied Informatics theme to highlight the crucial application and implementation focus of their work. This is first time the Informatics Summit convenes outside of San Francisco. We are confident Chicago will bring new collaborations and connections. We look forward to receiving your submissions"

    query_scores = getDoc2VecScores(dataClean.preprocess_sentences, query, d2v_model)

    recs = getDoc2VecRecs(query_scores, 10, wikicfp)
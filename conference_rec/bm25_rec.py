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

# %% (Initialize nltk and lemmatizer)
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("stopwords")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

verb_codes = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}

# %% (read in data)
wikicfp = dataClean.read_folder("/workspaces/VRA/conference_rec/wikicfp_csv")
wikicfp = dataClean.unique_confs_per_year(wikicfp)
wikicfp = dataClean.better_dates(wikicfp)

# %% (pre-process data)
wiki_token = wikicfp.copy()
wiki_token = wiki_token[["Conference Title", "WikiCFP Tags", "Conference Description"]]
wiki_token.columns = ["title", "tags", "description"]
wiki_token.fillna('', inplace = True)
wiki_token["soup"] = wiki_token["tags"] + " " + wiki_token["description"]

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

with mp.Pool(mp.cpu_count()) as pool:
    wiki_token["processed_soup"] = pool.map(preprocess_sentences, wiki_token["soup"])

# %% (create bm25 class)
wiki_bm = wiki_token["processed_soup"].copy().to_list()
wiki_bm_token = [doc.split(" ") for doc in wiki_bm]

bm25 = bm(wiki_bm_token)

# %% (run query)
query = "AMIA 2022 Informatics Summit: From discovering innovative methods to learning from exciting real-world applications, AMIA 2022 Informatics Summit attendees will experience the full range of cutting-edge work in translational informatics and clinical data science from inception to implementation. This conference is the ideal setting for researchers, educators, data scientists, software developers and analysts, students, and industry professionals. The size of the conference makes it ideal for developing meaningful new connections and partnerships while learning practical advice to solve real-world challenges. New to the AMIA 2022 Informatics Summit, we have expanded upon the previous Informatics Implementation track to include it as a new theme: Applied Informatics. In addition to selecting one of the three core Programmatic Tracks (Clinical Research Informatics, Data Science, Translational Bioinformatics), authors/presenters can also choose to designate their submission as part of the Applied Informatics theme to highlight the crucial application and implementation focus of their work. This is first time the Informatics Summit convenes outside of San Francisco. We are confident Chicago will bring new collaborations and connections. We look forward to receiving your submissions"
query = preprocess_sentences(query)

token_query = query.split(" ")

doc_scores = bm25.get_scores(token_query)

# %% (retrieve results)
results = np.argsort(doc_scores)[-10:]
results = np.flip(results)

final_recs = wikicfp.loc[wikicfp.index[results]]
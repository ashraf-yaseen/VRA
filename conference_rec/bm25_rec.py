# %%
import nltk
import glob

import pandas as pd
import numpy as np

import data_cleaning as dataClean

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

wiki_token["processed_soup"] = wiki_token["soup"].apply(preprocess_sentences)

# %% (create bm25 class)
wiki_bm = wiki_token["processed_soup"].copy().to_list()
wiki_bm_token = [doc.split(" ") for doc in wiki_bm]

bm25 = bm(wiki_bm_token)

# %% (run query)
query = "CLOUD 2021 : 10th International Conference on Cloud Computing: Services and Architecture"
query = preprocess_sentences(query)

token_query = query.split(" ")

doc_scores = bm25.get_scores(token_query)

# %% (retrieve results)
results = np.argsort(doc_scores)[-10:]
results = np.flip(results)

final_recs = wikicfp.loc[wikicfp.index[results]]
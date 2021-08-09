#%%
import glob
from io import BytesIO
import nltk

import multiprocessing as mp
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#%%
def readFolder(path: str) -> pd.DataFrame:
    """
    Reads all given csv files in a directory and outputs a dataframe of all csv files concatenated together

    Input Args: Path to directory where csv files are stored
    Output: Dataframe containing all data from all input csv files
    """
    files = glob.glob(f"{path}/*.csv") 
    
    dfs = [pd.read_csv(a_file, index_col = 0, header = 0) for a_file in files]

    collective_df = pd.concat(dfs, ignore_index = True)

    return collective_df

def readFolderST(input_files: BytesIO) -> pd.DataFrame:
    """
    Reads all given csv files in a directory and outputs a dataframe of all csv files concatenated together

    Input Args: all relevant files in BytesIO format for streaming
    Output: Dataframe containing all data from all input csv files
    """
    for input_file in input_files:
        input_file.seek(0)

    dfs = [pd.read_csv(input_file, index_col = 0, header = 0) for input_file in input_files]
    
    collective_df = pd.concat(dfs, ignore_index = True)

    return collective_df

def betterDates(df: pd.DataFrame, date_column: str = "Conference Date") -> pd.DataFrame:
    """
    Takes date column and creates two new columns: start date and end date in datetime format for easy usage

    Input Args: Dataframe, Date column of dataframe
    Output: Start Date and End Date datetime columns in a dataframe
    """
    df[["startDate", "endDate"]] = df[date_column].str.split("-", expand = True)
    
    df["endDate"] = pd.to_datetime(df["endDate"])
    df["startDate"] = pd.to_datetime(df["startDate"])

    return df

def uniqueConfsPerYear(df: pd.DataFrame) -> pd.DataFrame:
    """

    Takes input dataframe and  drops all non-unique entries per each year

    Input: Dataframe
    Output: Dataframe
    """
    NU_frame1 = df.copy()
    NU_frame1["Year"] = NU_frame1["Conference Title"].str.extract(r"(\d{4}) :")
    NU_frame1["Year"] = NU_frame1["Year"].apply(int)
    U_frame = NU_frame1.drop_duplicates(subset = "Conference Title")

    return U_frame

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

def preprocess_sentences(text, lemma_func):
    stop_words = set(stopwords.words("english"))
    verb_codes = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}

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

def multiprocessApply(function, corpus, data_col, new_col, lemma_func):
    with mp.Pool(mp.cpu_count()) as pool:
        corpus[new_col] = pool.map(function, corpus[data_col], lemma_func)

    return corpus

# %%
if __name__ == "__main__":
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("wordnet")
    nltk.download("stopwords")

    lemmatizer = setLemmatizer()

    wikicfp = readFolder("/workspaces/VRA/conference_rec/wikicfp_csv")
    wikicfp = uniqueConfsPerYear(wikicfp)
    wikicfp = betterDates(wikicfp)

    wiki_token = processCorpus(wikicfp)

    wiki_token = multiprocessApply(preprocess_sentences, wiki_token, "soup", "processed_soup", lemma_func = lemmatizer)
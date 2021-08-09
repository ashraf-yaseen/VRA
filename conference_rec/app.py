import gensim.models
import nltk
import streamlit as st

import multiprocessing as mp
import numpy as np
import pandas as pd

from io import StringIO, BytesIO
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi as bm

st.set_page_config("Conference Recommendations", None, layout = "wide")
st.title("Conference Recommendations")
st.sidebar.title("Recommender Options")

# define functions locally
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

def preprocess_sentences(text):
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

@st.cache(suppress_st_warning = True)
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

def createDoc2VecObject(corpus, data_col):
    wiki_gen = corpus[data_col].copy().to_list()
    wiki_gen_token = [doc.split(" ") for doc in wiki_gen]

    wiki_gensim = [gensim.models.doc2vec.TaggedDocument(d, [i]) for i, d in enumerate(wiki_gen_token)]

    return wiki_gensim

@st.cache(suppress_st_warning = True, allow_output_mutation=True)
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

# pre download nltk data and set stop words + verb codes
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find("averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger")
try:
    nltk.data.find("wordnet")
except LookupError:
    nltk.download("wordnet")
try:
    nltk.data.find("stopwords")
except LookupError:
    nltk.download("stopwords")

lemmatizer = setLemmatizer()

# get corpus from user to be used for the recommendations
input_files = st.sidebar.file_uploader("Choose your corpus", accept_multiple_files = True)

if input_files:
    wikicfp = readFolderST(input_files)
    wikicfp = uniqueConfsPerYear(wikicfp)
    wikicfp = betterDates(wikicfp)
    st.write("The following is your raw corpus:")
    st.dataframe(wikicfp)
else:
    st.write("You have not created a corpus yet")
    st.stop()

# process corpus to get it into the list of lists format used by BM25 and Doc2Vec
if wikicfp is not None:
    wiki_token = processCorpus(wikicfp)
    wiki_token = multiprocessApply(preprocess_sentences, wiki_token, "soup", "processed_soup")
    st.write("The following is a sample of your tokenized corpus:")
    st.dataframe(wiki_token.head(1000), 1080)

# define batch element for choosing which type of recommender to use
with st.sidebar.form(key = "form_1"):
    rec_type = st.radio("Choose a recommender", ("BM25", "Doc2Vec"))
    query_type = st.radio("Choose query format", ("File", "Textbox"))
    number_of_recs = st.number_input("How many reccomendations would you like", 5, 50, value = 10, step = 5) 
    submit_button = st.form_submit_button(label = "Submit")

# create recommendations based on recommender algorithm and input type
if rec_type == "BM25" and query_type == "Textbox":
    query = st.sidebar.text_area("Enter your query", '', help = "Please enter the query you want conference recommendations for")
    if not query: 
        st.write("Please enter the query you want conference recommendations for")
        st.stop()
    bm25_model = createBMObject(wiki_token, "processed_soup")
    query_scores = getBM25Ranks(preprocess_sentences, query, bm25_model)
    recs = getRecs(query_scores, number_of_recs, wikicfp)
    st.write(f"Here are the top {number_of_recs} recommendations for your query 🎉:")
    st.table(recs[["Conference Title", "Conference Webpage"]])
    st.stop()
elif rec_type == "BM25" and query_type == "File":
    query = st.sidebar.file_uploader("Choose your query", type = ["txt"], help = "Please select the file you want conference recommendations for (must be a txt file)")
    if query:
        stringio = StringIO(query.getvalue().decode("utf-8"))
        query_string = stringio.read()
    else: 
        st.write("Please enter the query you want conference recommendations for")
        st.stop()
    bm25_model = createBMObject(wiki_token, "processed_soup")
    query_scores = getBM25Ranks(preprocess_sentences, query_string, bm25_model)
    recs = getRecs(query_scores, number_of_recs, wikicfp)
    st.write(f"Here are the top {number_of_recs} recommendations for your query 🎉:")
    st.table(recs[["Conference Title", "Conference Webpage"]])
    st.stop()
elif rec_type == "Doc2Vec" and query_type == "Textbox":
    query = st.sidebar.text_area("Enter your query", '', help = "Please enter the query you want conference recommendations for")
    if not query: 
        st.write("Please enter the query you want conference recommendations for")
        st.stop()
    d2v_corpus = createDoc2VecObject(wiki_token, "processed_soup")
    d2v_model = createModel(d2v_corpus)
    query_scores = getDoc2VecScores(preprocess_sentences, query, d2v_model)
    recs = getDoc2VecRecs(query_scores, number_of_recs, wikicfp)
    st.write(f"Here are the top {number_of_recs} recommendations for your query 🎉:")
    st.table(recs[["Conference Title", "Conference Webpage"]])
    st.stop()
elif rec_type == "Doc2Vec" and query_type == "File":
    query = st.sidebar.file_uploader("Choose your query", type = ["txt"], help = "Please select the file you want conference recommendations for (must be a txt file)")
    if query:
        stringio = StringIO(query.getvalue().decode("utf-8"))
        query_string = stringio.read()
    else: 
        st.write("Please enter the query you want conference recommendations for")
        st.stop()
    d2v_corpus = createDoc2VecObject(wiki_token, "processed_soup")
    d2v_model = createModel(d2v_corpus)
    query_scores = getDoc2VecScores(preprocess_sentences, query_string, d2v_model)
    recs = getDoc2VecRecs(query_scores, number_of_recs, wikicfp)
    st.write(f"Here are the top {number_of_recs} recommendations for your query 🎉:")
    st.table(recs[["Conference Title", "Conference Webpage"]])
    st.stop()
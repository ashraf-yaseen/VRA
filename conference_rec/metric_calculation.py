from datetime import datetime

import glob
import pytz

import pandas as pd
import numpy as np


def precision_k(df, k):
    
    if len(df) > k:
        ratings = df["User Rating"][:k]
    else:
        ratings = df["User Rating"]

    relevant = 0

    for item in ratings:
        if item >= 3:
            relevant += 1
    
    precision_score = relevant/k

    return precision_score


def average_precision_k(df, k):
    
    def precision_k(df, k):
        
        if len(df) > k:
            ratings = df["User Rating"][:k]
        else:
            ratings = df["User Rating"]

        relevant = 0

        for item in ratings:
            if item >= 3:
                relevant += 1
        
        precision_score = relevant/k

        return precision_score

    precision_list = []

    for i in range(1, k+1, 1):
        precision_value = precision_k(df, i)
        precision_list.append(precision_value)
    
    avg_precision_df = df.copy()
    avg_precision_df["precision@i"] = precision_list

    avg_precision_df = avg_precision_df[avg_precision_df["User Rating"] >= 3]

    avg_precision_score = np.mean(avg_precision_df["precision@i"])

    return avg_precision_score

def readFolder(path):
    files = glob.glob(f"{path}/*.csv") 
    
    dfs = [pd.read_csv(a_file, index_col = 0, header = 0) for a_file in files]

    return dfs

def filename(path= "/workspaces/VRA/conference_rec/app_ratings/"):
    today = datetime.now(pytz.timezone("US/Central")).date()
    excel_name = f"{path}conference_recommender_evaluations_{today}.xlsx"

    return excel_name

if __name__ == "__main__":
    ### DON'T FORGET EVERYTHING IS SORTED IN THE ORDER IT APPEARS IN THE DIRECTORY

    bm25_ratings = readFolder("/workspaces/VRA/conference_rec/app_ratings/bm25")
    doc2vec_ratings = readFolder("/workspaces/VRA/conference_rec/app_ratings/doc2vec")
    tfidf_ratings = readFolder("/workspaces/VRA/conference_rec/app_ratings/tfidf")

    bm_25_precision = []
    bm_25_avg_precision = []
    doc_2_vec_precision = []
    doc_2_vec_avg_precision = []
    tfidf_precision = []
    tfidf_avg_precision = []

    for df in bm25_ratings:
        bm25_df_precision = precision_k(df, 10)
        bm25_df_avg_precision = average_precision_k(df, 10)
        bm_25_precision.append(bm25_df_precision)
        bm_25_avg_precision.append(bm25_df_avg_precision)

    for df in doc2vec_ratings:
        doc_2_vec_df_precision = precision_k(df, 10)
        doc_2_vec_df_avg_precision = average_precision_k(df, 10)
        doc_2_vec_precision.append(doc_2_vec_df_precision)
        doc_2_vec_avg_precision.append(doc_2_vec_df_avg_precision)

    for df in tfidf_ratings:
        tfidf_df_precision = precision_k(df, 10)
        tfidf_df_avg_precision = average_precision_k(df, 10)
        tfidf_precision.append(tfidf_df_precision)
        tfidf_avg_precision.append(tfidf_df_avg_precision)

    bm_25_mean_average_precision = np.mean(bm_25_avg_precision)
    doc_2_vec_mean_average_precision = np.mean(doc_2_vec_avg_precision)
    tfidf_mean_average_precision = np.mean(tfidf_avg_precision)

    excel_name = filename()

    with pd.ExcelWriter(excel_name) as writer:
        pd.Series(bm_25_precision).to_excel(writer, sheet_name = "BM25 Precision@k", index = False)
        pd.Series(bm_25_avg_precision).to_excel(writer, sheet_name = "BM25 AP@k", index = False)
        pd.Series(bm_25_mean_average_precision).to_excel(writer, sheet_name = "BM25 Mean AP@k", index = False)
        pd.Series(doc_2_vec_precision).to_excel(writer, sheet_name = "Doc2Vec Precision@k", index = False)
        pd.Series(doc_2_vec_avg_precision).to_excel(writer, sheet_name = "Doc2Vec AP@k", index = False)
        pd.Series(doc_2_vec_mean_average_precision).to_excel(writer, sheet_name = "Doc2Vec Mean AP@k", index = False)
        pd.Series(tfidf_precision).to_excel(writer, sheet_name = "TFIDF Precision@k", index = False)
        pd.Series(tfidf_avg_precision).to_excel(writer, sheet_name = "TFIDF AP@k", index = False)
        pd.Series(tfidf_mean_average_precision).to_excel(writer, sheet_name = "TFIDF Mean AP@k", index = False)
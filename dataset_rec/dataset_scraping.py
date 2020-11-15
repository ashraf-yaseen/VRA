# %%
import time

import numpy as np
import pandas as pd

from Bio import Entrez

# %%
def extract_title(citations):
    citations_list = list(citations.copy().apply(str))
    title = []
    for i in range(0,len(citations_list)):
        title.append(citations_list[i].split(".",1)[0])

    return title

def get_pmid(ref_title, contact="rachit.sabharwal@uth.tmc.edu", key=""):
    ''' Using the Entrez search term, it queries the eSearch endpoint of the Entrez api to retrieve the corresponding pmids'''
    pmid = []
    counter = 0
    for i in range(len(ref_title)):
        if "PMID" not in ref_title:
            Entrez.email = contact
            Entrez.api_key = key
            handle = Entrez.esearch(db='pubmed', term = ref_title[i], retmax=1)
            record = Entrez.read(handle)
            pmid.append(record['IdList'])

            if counter == 5:
                time.sleep(60)
                counter = 0
            
            counter += 1

    return pmid

def get_doi(ref_pmid, contact="rachit.sabharwal@uth.tmc.edu", key=""):
    ''' Using the pmids, it queries the eSummary endpoint to retrieve the corresponding dois and join them to the input df. ''' 
    doi = []
    counter = 0
    for i in range(len(ref_pmid)):
        if not ref_pmid[i] == []:
            Entrez.email = contact
            Entrez.api_key = key
            handle = Entrez.esummary(db='pubmed', id=ref_pmid[i], retmax=1)
            record = Entrez.read(handle)
            info = record[0]['ArticleIds']
            doi.append(info)

            if counter == 5:
                time.sleep(60)
                counter = 0
            
            counter += 1
    
    return doi

# %%
if __name__ == "__main__":
    array_express = pd.read_csv("array_express_20181217.csv")
    ae_citations = array_express[["citations", "citations_url"]]
    citations = ae_citations["citations"].copy().dropna().drop_duplicates()
    title = extract_title(citations)
    top_twofifty = title[0:4]
    top_250_pmids = get_pmid(top_twofifty)
    top_250_dois = get_doi(top_250_pmids)
    top_250_pmids_df = pd.DataFrame(top_250_pmids)
    top_250_dois_df = pd.DataFrame(top_250_dois)
    top_250_pmids_df.to_csv('top_250_pmids.csv')
    top_250_dois_df.to_csv('top_250_dois.csv')
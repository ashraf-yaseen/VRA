# %%
import requests
import selenium

import pandas as pd

from bs4 import BeautifulSoup

# %%
# Load RFA files into memory
rfa_list = pd.read_pickle(r"G:\My Drive\Backup\Documents\UTHSC Biostats\Recommender Systems\Grant Recommendations\rfa_ls.ls")
# %% 
# Specify base url (URL for RFA to be used while scraping)
base = 'https://grants.nih.gov/grants/guide/rfa-files/'

# Create individual links per RFA
# Using just 1 RFA for testing purposes for now
url = base + rfa_list[0] + '.html'
# %% 
# Use BeautifulSoup to scrape the contents of the webpage
# TODO (FIX) Only the bottom third of the page is being scraped currently!
r = requests.get(url, headers = {'user-agent': 'Mozilla/5.0'})
soup = BeautifulSoup(r.content, "lxml")
# %%
# Use pandas to scrape webpage (only information in tabular form is scraped)
rfa_table = pd.read_html(url)
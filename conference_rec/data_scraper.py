# %%
import requests
import selenium

import pandas as pd
import feedparser as fp

from bs4 import BeautifulSoup

# %%
# Specify base url 
base = r'http://www.wikicfp.com/cfp/call?conference=machine%20learning&skip=1'

# Using just 1 category for testing purposes for now
url = base

# %% 
# Use BeautifulSoup to scrape the contents of the webpage
# TODO (FIX) Only the bottom third of the page is being scraped currently!
r = requests.get(url, headers = {'user-agent': 'Mozilla/5.0'})
soup = BeautifulSoup(r.content, "lxml")

# %%
# Use pandas to scrape webpage (only information in tabular form is scraped)
rfa_table = pd.read_html(url)
# %%

feed = fp.parse("https://conferencemonkey.org/rss")
entry = feed.entries[0]
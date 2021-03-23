# %%
import requests

import pandas as pd

from bs4 import BeautifulSoup
from chromedriver_py import binary_path
from selenium import webdriver

# %%
# website 1 is www.wikicfp.com
base_url = "www.wikicfp.com"

# initiate selenium webdriver
driver = webdriver.Chrome(executable_path = binary_path)
driver.get(base_url)
# %%

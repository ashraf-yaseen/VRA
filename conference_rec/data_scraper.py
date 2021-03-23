# %%
import requests

import pandas as pd

from bs4 import BeautifulSoup
from chromedriver_py import binary_path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# %%
# website 1 is www.wikicfp.com
base_url = "www.wikicfp.com"

# initiate selenium webdriver
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(executable_path = binary_path, chrome_options = chrome_options)
driver.get(base_url)
# %%

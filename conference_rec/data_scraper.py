# %%
import requests

import pandas as pd

from bs4 import BeautifulSoup
from chromedriver_py import binary_path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# %%
# website 1 is www.wikicfp.com
base_url = r"http://www.wikicfp.com/cfp/call?conference=public%20health"

# initiate selenium webdriver
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(executable_path = binary_path, options = chrome_options)
driver.get(base_url)
# %%
# TODO: these need to be functions 
all_links = driver.find_elements_by_xpath("//a[contains(@href, 'event.showcfp?')]")
for link in all_links:
    print(link.get_attribute("href"))

pages = driver.find_elements_by_xpath("//td[contains(text(), 'Total of')]")
print([page.text for page in pages])

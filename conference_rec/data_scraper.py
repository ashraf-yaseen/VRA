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
# TODO: these need to be functions?
last_page = driver.find_element_by_link_text("last")
last_page_num = last_page.get_attribute("href")
last_page_num = int(last_page_num[len(last_page_num) - 1])

page_num = 1
url_list = []
while page_num < last_page_num:
    all_links = driver.find_elements_by_xpath("//a[contains(@href, 'event.showcfp?')]")
    for link in all_links:
        url_list.append(link.get_attribute("href"))
    
    next_page = driver.find_element_by_link_text("next")
    next_page_num = next_page.get_attribute("href")
    next_page_num = int(next_page_num[len(next_page_num) - 1])

    if next_page_num > page_num:
        next_page.click()
        page_num += 1
# %%

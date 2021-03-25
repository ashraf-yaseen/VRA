# %%
import pandas as pd

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
conference_title = []
conference_link = []
location = []
date = []
conference_description = []
tags = []

for url in url_list:
    driver.get(url)
    try:
        url_title = driver.find_element_by_css_selector("head > meta:nth-child(3)").get_attribute("content")
    except:
        url_title - "N/A"
    
    try:
        url_link = driver.find_element_by_css_selector("body > div:nth-child(5) > center > table > tbody > tr:nth-child(3) > td > a").get_attribute("href")
    except:
        url_link = "N/A"
    
    try:
        url_date = driver.find_element_by_css_selector("body > div:nth-child(5) > center > table > tbody > tr:nth-child(5) > td > table > tbody > tr > td > table > tbody > tr:nth-child(1) > td > table > tbody > tr:nth-child(1) > td").text
    except: 
        url_date = "N/A"
    
    try:
        url_location = driver.find_element_by_css_selector("body > div:nth-child(5) > center > table > tbody > tr:nth-child(5) > td > table > tbody > tr > td > table > tbody > tr:nth-child(1) > td > table > tbody > tr:nth-child(2) > td").text
    except:
        url_location = "N/A"
    
    try:
        url_tags = driver.find_element_by_css_selector("body > div:nth-child(5) > center > table > tbody > tr:nth-child(5) > td > table > tbody > tr > td > table > tbody > tr:nth-child(2) > td > table > tbody > tr:nth-child(2) > td > h5").text.replace("Categories", "")
    except:
        try:
            url_tags = driver.find_element_by_css_selector("body > div:nth-child(5) > center > table > tbody > tr:nth-child(4) > td > table > tbody > tr > td > table > tbody > tr:nth-child(2) > td > table > tbody > tr:nth-child(2) > td > h5").text.replace("Categories", "")
        except:
            url_tags = "N/A"

    try:
        url_description = driver.find_element_by_css_selector("body > div:nth-child(5) > center > table > tbody > tr:nth-child(8) > td > div").text.replace("\n", " ")
    except:
        try:
            url_description = driver.find_element_by_css_selector("body > div:nth-child(5) > center > table > tbody > tr:nth-child(7) > td > div").text.replace("\n", " ")
        except:
            url_description = "N/A"

    conference_title.append(url_title)
    conference_link.append(url_link)
    location.append(url_location)
    date.append(url_date)
    conference_description.append(url_description)
    tags.append(url_tags)

driver.quit()
# %%
wikicfp_publichealth = pd.DataFrame(zip(conference_title, 
                                        conference_link,
                                        date,
                                        location,
                                        tags,
                                        url_list,
                                        conference_description),
                                        columns = ["Conference Title",
                                                    "Conference Webpage",
                                                    "Conference Date",
                                                    "Conference Location",
                                                    "WikiCFP Tags",
                                                    "WikiCFP Link",
                                                    "Conference Description"])

# %%
wikicfp_publichealth.to_excel("wikicfp_publichealth.xlsx")
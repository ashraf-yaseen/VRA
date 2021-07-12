# %%
import time
import logging

import pandas as pd
import numpy as np
import sqlalchemy as sql

from chromedriver_py import binary_path
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.chrome.options import Options

# %%
def get_category_cfp_links(url, driver):
    driver.get(url)

    try:
        next_page_link = driver.find_element_by_link_text("next")
        next_page = next_page_link.get_attribute("href")
    except NoSuchElementException:
        pass

    last_page_link = driver.find_element_by_link_text("last")
    last_page = last_page_link.get_attribute("href")

    current_page = driver.current_url

    url_list = []
    while current_page != next_page:
        all_links = driver.find_elements_by_xpath("//a[contains(@href, 'event.showcfp?')]")
        for link in all_links:
            url_list.append(link.get_attribute("href"))
        
        try:
            next_page_link = driver.find_element_by_link_text("next")
            next_page = next_page_link.get_attribute("href")
        except NoSuchElementException:
            pass

        current_page = driver.current_url

        try: 
            next_page_link.click()
        except StaleElementReferenceException:
            break

    return url_list

def scrape(url_list, driver):
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
            url_title = "N/A"
        
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

    wikicfp = pd.DataFrame(zip(conference_title, 
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

    return wikicfp

def create_cat_urls(categories):
    base = r"http://www.wikicfp.com/cfp/call?conference="

    cat_list = [base + category for category in categories]

    return cat_list

def sqlite_out(info_dict):
    engine = sql.create_engine('sqlite:///wikicfp.db', echo=False)

    for key, value in info_dict.items():
        value.to_sql(key, con=engine, if_exists='replace')

# %%
if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    categories = ["NLP", "bioinformatics", "environment", "healthcare", "biotechnology", "renewable%20energy", "natural%20language%20processing", "biology", "biomedical%20engineering", "data%20science", "medicine", "environmental%20engineering", "chemistry", "informatics", "medical",
    "information%20science", "informatics", "sustainable%20development", "biomedical", "health%20informatics", "computational%20biology", "neuroscience", "cognitive%20science", "statistics", "life%20sciences", "nursing", "information%20system", "mobility", "medical%20imaging", "data%20analytics", "ehealth",
    "text%20mining", "chemical", "e-health", "public%20health", "chemical%20engineering", "analytics", "nutrition", "environmental%20sciences", "business%20intelligence", "recommender%20systems", "pediatrics", "ecology", "molecular%20biology", "cardiology", "cancer", "climate%20change", "environmental", "neurology",
    "life%20science", "oncology", "green%20computing", "biological%20sciences"]
    
    names = ["NLP", "bioinformatics", "environment", "healthcare", "biotechnology", "renewable-energy", "natural-language-processing", "biology", "biomedical-engineering", "data-science", "medicine", "environmental-engineering", "chemistry", "informatics", "medical",
    "information-science", "informatics", "sustainable-development", "biomedical", "health-informatics", "computational-biology", "neuroscience", "cognitive-science", "statistics", "life-sciences", "nursing", "information-system", "mobility", "medical-imaging", "data-analytics", "ehealth",
    "text-mining", "chemical", "e-health", "public-health", "chemical-engineering", "analytics", "nutrition", "environmental-sciences", "business-intelligence", "recommender-systems", "pediatrics", "ecology", "molecular-biology", "cardiology", "cancer", "climate-change", "environmental", "neurology",
    "life-science", "oncology", "green-computing", "biological-sciences"]

    cat_urls = create_cat_urls(categories)

    # intitiate selenium webdriver
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(executable_path = binary_path, options = chrome_options)

    wikicfp_info = {}

    for index, item in enumerate(cat_urls):
        try:
            url_list = get_category_cfp_links(item, driver)
            wikicfp = scrape(url_list, driver)
            wikicfp_info[f"wikicfp_{names[index]}"] = wikicfp
            time.sleep(30)
        except:
            logger.exception(f"Failed to scrape {names[index]}")

            continue
    
    sqlite_out(wikicfp_info)
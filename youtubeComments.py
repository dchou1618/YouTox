#!/usr/bin/env python3

# miscellaneous
import sys
import time
#import youtubesentiment
# dynamic web scraping
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options = webdriver.ChromeOptions()
options.add_argument('headless')
# static webscraping
from bs4 import BeautifulSoup
# sentiment
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# plotting
import matplotlib.pyplot as plt


class YouTubeComments:
    chromeDriver = webdriver.Chrome(r"./chromedriver.exe",
                                    options = options)
    def __init__(self,**kwargs):
        for key in kwargs:
            val = kwargs[key]
            if key == "url":
                exec("self.{} = \"{}\"".format(key,val))
            else:
                exec("self.{} = {}".format(key,val))
    def scrape(self):
        """
        Scrapes youtube comments
        """
        def scroll():
            """scrolls through youtube page"""
            for i in range(0,self.bottom,self.increment):
                prompt = 'window.scrollTo({}, {});'.format(i,i+self.increment)
                YouTubeComments.chromeDriver.execute_script(prompt)
                time.sleep(1)

        YouTubeComments.chromeDriver.get(self.url)
        scroll()
        ytHTML = BeautifulSoup(YouTubeComments.chromeDriver.page_source,"lxml")

        wait = WebDriverWait(YouTubeComments.chromeDriver, 5)
        comments = []
        for comment in wait.until(EC.presence_of_all_elements_located(\
                          (By.CSS_SELECTOR, "#comment #content-text"))):
            comments.append(comment.text)
        self.comments = comments

def runCommentScrape(url="https://www.youtube.com/watch?v=OtRuEhm9Eoo"):
    try:
        ytComments = YouTubeComments(url=url,bottom = 3000,increment = 350)
        ytComments.scrape()
        return ytComments.comments
    except Exception as e:
        return "Error"+str(e)

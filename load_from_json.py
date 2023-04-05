# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 13:05:59 2022

@author: kgeor
"""

import json
from newspaper import Article
import spacy
import pandas as pd

nlp = spacy.load("de_core_news_sm")

f = open('GermanFakeNC.json')

data = json.load(f)

datalist = []

rating = []

titles = [] 

for i in data:
    
    try:
        
        link = i["URL"]

        article = Article(link)

        article.download()

        article.parse()

        title = article.title

        doc = article.text

        datalist.append(doc)

        titles.append(title)

        rating.append(i['Overall_Rating'])
    
    except:
        continue;
        
        
f.close()


dataframe = pd.DataFrame(list(zip(titles,datalist, rating)),
              columns=['titles','text','label'])

dataframe.to_csv('fake_news.csv', index=False)

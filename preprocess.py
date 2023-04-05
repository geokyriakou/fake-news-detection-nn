# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 13:07:44 2022

@author: kgeor
"""

import re
import spacy
from spacy.lang.de.stop_words import STOP_WORDS
import pandas as pd
nlp = spacy.load("de_core_news_sm")    
import numpy as np


def clean(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ''.join(emoticons).replace('-', '')
    doc = nlp(text)
    text = ' '.join([token.lemma_ for token in doc if token.text not in STOP_WORDS])
    return text

def preprocessor(text,label,title = None):
    
    if(title is not None):
        text = title + text
        
    dataframe = pd.DataFrame(list(zip(text,label)),
              columns=['text','label'])
    
    dataframe = dataframe[dataframe['label'] != 0.5]
    
    label = dataframe.label
    
    text = dataframe.text
    
    label = np.where(label > 0.5,1, 0)
    
    text = text.apply(lambda x : clean(x))
    
    return (text,label)

def preprocess(path,name_text,name_label,name_title = None):
    df = pd.read_csv(path)
    
    df.dropna(inplace=True)

    df.reset_index(drop=True, inplace=True)
    
    return preprocessor(df[name_text],df[name_label],df[name_title])
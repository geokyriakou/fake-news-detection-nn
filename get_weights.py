# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 21:36:21 2022

@author: kgeor
"""
from keras.models import load_model
model = load_model('best_model')
print(model.get_weights())

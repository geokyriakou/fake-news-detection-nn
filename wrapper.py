# -*- coding: utf-8 -*-
"""
To test with your own data just modify the test function call on the main
function (the last function).The first parameter is the path,the second is 
the name of the column that contains the article,the second one is the label
and the final one is titles with is optional.In this example we insert our 
fake_news.csv file which contains this columns (text,label,titles).

@author: kgeor
"""


#To test with your 
from keras.preprocessing.text import text_to_word_sequence
from keras.models import load_model
from keras.preprocessing import sequence
from sklearn.metrics import classification_report
import pickle
from preprocess import preprocess

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    

#---------------------- HELPER FUNCTIONS -----------------------------#
def get_word_index(word):

    try:
        return tokenizer.word_index[word]
    except:
        return None

def text_to_int_sequence(text):
    seq = [get_word_index(word) for word in text_to_word_sequence(text)]
    return [index for index in seq if index]


def custom_fit(model,validation_data,X,y,callback,max_seq_length):
    
    X = X.apply(text_to_int_sequence)

    X_pad = sequence.pad_sequences(X, maxlen=max_seq_length)

    X_valid = validation_data[0].apply(text_to_int_sequence)
        
    X_valid_pad = sequence.pad_sequences(X_valid, maxlen=max_seq_length)

    y_valid = validation_data[1]
    
    model.fit(X_pad, y, validation_data=(X_valid_pad, y_valid), 
                  epochs=200, batch_size=256,callbacks = callback)
    

def custom_predict(model,X,max_seq_length):
    X = X.apply(text_to_int_sequence)
    X = sequence.pad_sequences(X, maxlen = max_seq_length)
    
    return (model.predict(X) > 0.5).astype("int32")


def test(path,name_text,name_label,name_title):
    X,y = preprocess(path,name_text,name_label,name_title)

    model = load_model('best_model')
    
    y_pred = custom_predict(model,X,max_seq_length=512)

    print(classification_report(y, y_pred))

        
#-------------------------------MAIN ---------------------------------#

def main():

    test("fake_news.csv","text","label","titles")
    

if __name__ == "__main__":
    main()
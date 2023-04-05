# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 13:08:16 2022

@author: kgeor
"""

from keras.preprocessing.text import text_to_word_sequence
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
from preprocess import preprocess
import timeit


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

#-------------------------------MAIN ---------------------------------#

X,y = preprocess("fake_news.csv","text","label","titles")

tokenizer = Tokenizer()

tokenizer.fit_on_texts(X)

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


max_vocab = len(tokenizer.word_index) + 20

max_seq_length=512

embedding_vector_length = 2*150

model = Sequential()

model.add(Embedding(max_vocab, embedding_vector_length, input_length=max_seq_length))

model.add(Conv1D(filters=2*1*32, kernel_size=5, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=2*2*32, kernel_size=5, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(32))

model.add(Dropout(0.25))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_accuracy',
                      min_delta=0,
                      patience=30,
                      verbose=2, mode='max')

checkpoint = ModelCheckpoint(filepath='best_model',
                             monitor='val_accuracy',
                             mode='max',
                             save_best_only=True)

callbacks_list = [early_stopping, checkpoint]

start = timeit.default_timer()
custom_fit(model,(X_valid, y_valid),X_train,y_train,callbacks_list,max_seq_length)
stop = timeit.default_timer()
print('Time training: ', stop - start)


model = load_model('best_model')
model.summary()

start = timeit.default_timer()
y_pred = custom_predict(model,X_test,max_seq_length)
stop = timeit.default_timer()
print('Time testing: ', stop - start)

print(classification_report(y_test, y_pred))
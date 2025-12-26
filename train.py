import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg
import pandas as pd

## Load the Gutenberg corpus
data=gutenberg.raw('shakespeare-hamlet.txt')

##save the data to a text file
with open('hamlet.txt', 'w', encoding='utf-8') as f:
    f.write(data)

###data preprocessing 
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

#load the dataset 
with open('hamlet.txt', 'r' )as file:
    text=file.read().lower()

#tokenization
tokenizer=Tokenizer()
tokenizer.fit_on_texts([text])
total_words=len(tokenizer.word_index)+1
print(total_words)

#create input sequence 
input_sequences=[]
for line in text.split('\n'):
    token_list=tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence=token_list[:i+1]
        input_sequences.append(n_gram_sequence)

##pad sequences 
max_sequence_len=max([len(x) for x in input_sequences])
print(max_sequence_len)
input_sequences=np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

#x and y split 
X=input_sequences[:,:-1]
y=input_sequences[:,-1]

import tensorflow as tf 
y=tf.keras.utils.to_categorical(y, num_classes=total_words)

#ttrain_test_split 
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

#train lstm rnn 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input

model=Sequential()
model.add(Input(shape=(max_sequence_len-1,)))
model.add(Embedding(total_words, 100))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss', patience=5)
history=model.fit(X_train, y_train, epochs=10, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stop])

# save the model 
model.save('lstm_text_generator.keras')

import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
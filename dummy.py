import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from numpy import asarray
from numpy import zeros
import re
import nltk
from imports import *
from keras import backend as K
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import GlobalMaxPooling1D, Convolution1D
from keras.layers.embeddings import Embedding
from keras.layers import Input
from keras import optimizers
from keras import regularizers
from keras.models import Model
import keras.layers.core
from keras.preprocessing.text import Tokenizer
from keras.layers import Conv2D
from keras.layers import MaxPooling1D
from keras.layers import Reshape
from keras.layers import Concatenate
from keras.layers import Flatten
from keras.layers.core import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.dummy import DummyClassifier



# Reading files
with open('SD.txt', 'r', errors='ignore') as f:
    negative = f.readlines()
with open('no_SD.txt', 'r', errors='ignore') as f:
    positive = f.readlines()

# Creating a dataFrame with two columns(sentence,label)
df = pd.DataFrame(columns=['comment', 'label'])
df['comment'] = negative + positive
df['label'] = [0] * len(negative) + [1] * len(positive)

# Shuffling the data in our dataFrame
df = df.sample(frac=1, random_state=50)
df.reset_index(inplace=True, drop=True)

# Preprocessing the data
LongestSent = 30
max_vocab_size = 20000

# Split the text into words
Word_array = []
for sentence in df['comment']:
    Word_array.append(text_to_word_sequence(sentence))

# Constructing the tokenizer, fit it, and convert the sequence of words to sequence of indexes
tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts([' '.join(x[:LongestSent]) for x in Word_array])
X = tokenizer.texts_to_sequences([' '.join(x[:LongestSent]) for x in Word_array])

# As we have sentences with different sizes, We need to do the padding
X = pad_sequences(X, maxlen=LongestSent, padding='post', truncating='post')
Y = df['label']

# splitting data to train and test set
X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
# splitting data to train and validation set
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=100)

baseline= DummyClassifier(strategy = 'most_frequent', random_state = 0, constant =None)
baseline.fit(X_train,y_train)
Mean_accuracy= baseline.score(X_test,y_test)
print("Most_frequent baseline accuracy:",Mean_accuracy)

baseline= DummyClassifier(strategy = 'stratified', random_state = 0, constant =None)
baseline.fit(X_train,y_train)
Mean_accuracy= baseline.score(X_test,y_test)
print("stratified baseline accuracy:",Mean_accuracy)

baseline= DummyClassifier(strategy = 'uniform', random_state = 0, constant =None)
baseline.fit(X_train,y_train)
Mean_accuracy= baseline.score(X_test,y_test)
print("uniform baseline accuracy:",Mean_accuracy)





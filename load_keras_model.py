from keras.models import load_model
import json
import nltk
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer #is based on The Porter Stemming Algorithm
from contractions import contractions_dict
from autocorrect import Speller
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn.model_selection import train_test_split
from numpy import asarray, array, zeros
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
# from keras.layers.embeddings import Embedding
# from keras.layers.recurrent import LSTM, Bidirectional
# from keras.layers.core import Dense
import matplotlib.pyplot as plt
import keras.metrics
from keras import backend as K
from keras import optimizers
from keras.layers import Dropout



def to_lower(text):
    """
    Converting text to lower case as in, converting "Hello" to  "hello" or "HELLO" to "hello".
    """
    # return ' '.join([w.lower() for w in nltk.word_tokenize(text)])
    lower_text = df['comments'].str.lower()

def strip_punctuation(text):
    """
    Removinmg puctuation
    """
    return ''.join(c for c in text if c not in punctuation)

def deEmojify(inputString):
    '''
    Removing emojis from text
    '''
    return inputString.encode('ascii', 'ignore').decode('ascii')


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def prep_text():

    #reading the labelled comment file
    with open('SD', 'r') as file:
        sd = file.readlines()

    with open('no_SD', 'r') as file:
        no_sd = file.readlines()
        # lower_text = []
        # data = json.load(json_file)
        # for i in data["memory_loss"]:
        #     lower_text.append(to_lower(i["comments"])) #converting the comments in memory_loss dictionary to lower case

    df = pd.DataFrame(columns=['comments', 'polarity'])
    df['comments'] = no_sd + sd
    df['polarity'] = [0] * len(no_sd) + [1] * len(sd)
    df = df.sample(frac=1, random_state = 10) #shuffling the rows
    df.reset_index(inplace = True, drop = True)
    df['comments'] = df['comments'].str.lower() #Converting text to lower case


    stopword = nltk.corpus.stopwords.words("english")
    not_stopwords = {'my', 'I', 'myself', 'me'} #removing some stopwords related to self-disclosure in nltk stopwords
    final_stop_words = set([word for word in stopword if word not in not_stopwords])
    speller = Speller()


    for i in range(len(df['comments'])):
        df['comments'][i] = re.sub("[0-9]+", " ", str(df['comments'][i])) #removing digits, since they're not important
        df['comments'][i] = deEmojify(df['comments'][i])
        df['comments'][i] = strip_punctuation(df['comments'][i])
        df['comments'][i] = ' '.join(speller(word) for word in df['comments'][i].split() if word not in final_stop_words) #removing stopwords and spell-correcting



    max_sent_len = 80
    max_vocab_size = 200
    word_seq = [text_to_word_sequence(comment) for comment in df['comments']]
    # print(word_seq)

    # vectorizing a text corpus, turning each text into either a sequence of integers (each integer being the index of a token in a dictionary)
    tokenizer = Tokenizer(num_words = max_vocab_size)
    tokenizer.fit_on_texts([' '.join(seq[:max_sent_len]) for seq in word_seq]) #Updates internal vocabulary based on a list of texts up to the max_sent_len.
    # print("vocab size: ", len(tokenizer.word_index)) #vocab size: 949

    #converting sequence of words to sequence of indices
    X = tokenizer.texts_to_sequences([' '.join(seq[:max_sent_len]) for seq in word_seq])
    X = pad_sequences(X, maxlen = max_sent_len, padding= 'post' , truncating='post')

    # y = df['polarity']
    # print(X)
    prediction = model.predict(np.array(X))
    print(prediction)
    df['polarity'] = prediction
    print(df['polarity'])
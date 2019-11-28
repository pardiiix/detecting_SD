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

def predict_label(file_name):

    #reading the labelled comment file
    with open(file_name, 'r') as file:
        comments = file.readlines()

        # lower_text = []
        # data = json.load(json_file)
        # for i in data["memory_loss"]:
        #     lower_text.append(to_lower(i["comments"])) #converting the comments in memory_loss dictionary to lower case

    df = pd.DataFrame(columns=['comments', 'exact_comments', 'polarity', 'confidence_score'])
    df['exact_comments'] = comments
    df['comments'] = comments
    df['comments'] = df['comments'].str.lower() #Converting text to lower case


    stopword = nltk.corpus.stopwords.words("english")
    not_stopwords = {'my', 'I', 'myself', 'me'} #removing some stopwords related to self-disclosure in nltk stopwords
    final_stop_words = set([word for word in stopword if word not in not_stopwords])
    speller = Speller()


    # df=df.drop(0 , axis = 0)
    for i in range(len(df['comments'])):
        if 'report' in df['comments'][i]:
            df = df.drop(i, axis=0)
        elif 'reply' in df['comments'][i]:
            df = df.drop(i, axis=0)
        elif 'likes' in df['comments'][i]:
            df = df.drop(i, axis=0)
        elif 'selection name' in df['comments'][i]:
            df = df.drop(i, axis=0)
        else:
            df['comments'][i] = re.sub("[0-9]+", " ", str(df['comments'][i])) #removing digits, since they're not important
            # df['comments'][i] = re.sub(r"reply", "", str(df['comments'][i])))
            # df['comments'][i] = re.sub(r"report", "", str(df['comments'][i]))
            df['comments'][i] = re.sub("[0-9]?likes", "", str(df['comments'][i]))
            df['comments'][i] = re.sub("[0-9]?replies", "", str(df['comments'][i]))
            df['comments'][i] = deEmojify(df['comments'][i])
            df['comments'][i] = strip_punctuation(df['comments'][i])
            df['comments'][i] = ' '.join(speller(word) for word in df['comments'][i].split() if word not in final_stop_words) #removing stopwords and spell-correcting (removed: )

    # print(df )

    max_sent_len = 200
    max_vocab_size = 4000
    word_seq = [text_to_word_sequence(comment) for comment in df['comments']]
    # print(word_seq)

    # vectorizing a text corpus, turning each text into either a sequence of integers (each integer being the index of a token in a dictionary)
    tokenizer = Tokenizer(num_words = max_vocab_size)
    tokenizer.fit_on_texts([' '.join(seq[:max_sent_len]) for seq in word_seq]) #Updates internal vocabulary based on a list of texts up to the max_sent_len.
    # print("vocab size: ", len(tokenizer.word_index)) #vocab size:

    #converting sequence of words to sequence of indices
    X = tokenizer.texts_to_sequences([' '.join(seq[:max_sent_len]) for seq in word_seq])
    X = pad_sequences(X, maxlen = max_sent_len, padding= 'post' , truncating='post')

    # y = df['polarity']
    # print(X)

    dependencies = {
        'recall_m': recall_m,
        'precision_m': precision_m,
        'f1_m': f1_m
    }

    model = load_model('saved_cnn_model.h5', custom_objects=dependencies)
    prediction = model.predict(np.array(X))
    df['confidence_score'] = prediction


    binary_prediction = np.where(prediction > 0.5, 1, 0)
    # print(prediction)
    df['polarity'] = binary_prediction
    # for i in range(len(df['polarity'])):
    #     if (df['polarity'].iloc[i] == 0):
    #         print(df['comments'].iloc[i])
    #         # print("h")
    # print(df['polarity'].value_counts())
    # print(df)
    # np.savetxt(r'/home/mo/pardis/new_labels.txt', df.values))

    print(df)

    for i in range(len(df['confidence_score'])):
    # for i in range(50):

        if 0.4 < df['confidence_score'].iloc[i] < 0.6:
            df['polarity'].iloc[i] = input("What is the label of this comment?:\n{}".format(df['exact_comments'].iloc[i]))
            # import ipdb;ipdb.set_trace()

    df.to_pickle("{}".format(str(file_name).split('/')[-1]))
    # print(df)
    return df

    # neg_df = pd.DataFrame(columns=['comments']) #creates a new empty dataframe
    # neg_df = df[df.polarity == 0] #adds negative comments to the dataframe
    # # print(neg_df['comments'])
    # neg_df['comments'].to_csv(path_or_buf='/home/mo/pardis/new_labels_neg.txt', mode='a', sep=' ', index=False, header=False) #saves negative comments to text

    # pos_df = pd.DataFrame(columns=['comments']) #creates a new empty dataframe (for positive examples)
    # pos_df = df[df.polarity == 1]
    # pos_df['comments'].to_csv(path_or_buf='/home/mo/pardis/new_labels_pos.txt', mode='a', sep=' ', index=False, header=False) #saves positive comments to text

    # print('new labels have been created for {}.'.format(file_name))

# predict_label('abdominal_comments.csv')
# predict_label('depression_comments.csv')
# predict_label('knee.csv')
import nltk
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer #is based on The Porter Stemming Algorithm
#from contractions import contractions_dict
from autocorrect import Speller
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn.model_selection import train_test_split
from numpy import asarray, array, zeros
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Convolution1D, MaxPooling1D, Flatten, Conv1D
import matplotlib.pyplot as plt
import keras.metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from keras import backend as K
from keras import optimizers
from keras.layers import Dropout
from keras.layers.core import Reshape
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



np.random.seed(500)
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


#reading the labelled comment file
with open('SD', 'r',errors='ignore') as file:
    sd = file.readlines()

with open('no_SD', 'r',errors='ignore') as file:
    no_sd = file.readlines()


df = pd.DataFrame(columns=['comments', 'polarity','text_final'])
df['comments'] = no_sd + sd
df['polarity'] = [0] * len(no_sd) + [1] * len(sd)
df = df.sample(frac=1, random_state = 10) #shuffling the rows
df.reset_index(inplace = True, drop = True)
df['comments'] = df['comments'].str.lower() #Converting text to lower case

# df['comments'].dropna(inplace=True)
# df['comments'] = [entry.lower() for entry in df['comments']]
# df['comments']= [word_tokenize(entry) for entry in df['comments']]

stopword = nltk.corpus.stopwords.words("english")
not_stopwords = {'my', 'I', 'myself', 'me', 'i'} #removing some stopwords related to self-disclosure in nltk stopwords
final_stop_words = set([word for word in stopword if word not in not_stopwords])
speller = Speller()

for i in range(len(df['comments'])):
    df['comments'][i] = re.sub("[0-9]+", " ", str(df['comments'][i])) #removing digits, since they're not important
    df['comments'][i] = deEmojify(df['comments'][i])
    df['comments'][i] = strip_punctuation(df['comments'][i])
    df['comments'][i] = ' '.join(speller(word) for word in df['comments'][i].split() if word not in final_stop_words) #removing stopwords and spell-correcting


X, X_test, y, y_test = model_selection.train_test_split(df['comments'],df['polarity'],random_state=10,test_size=0.2)
X_train, X_dev, y_train, y_dev = model_selection.train_test_split(X,y,random_state=10,test_size=0.2)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['comments'])
X_train_Tfidf = Tfidf_vect.transform(X_train)
X_dev_Tfidf = Tfidf_vect.transform(X_dev)
X_test_Tfidf = Tfidf_vect.transform(X_test)

#----------------------------------------------------------------
# Naive Bayes model
NB = naive_bayes.MultinomialNB()
NB.fit(X_train_Tfidf, y_train)
y_dev_pred = NB.predict(X_dev_Tfidf)
print("NB Validation set accuracy:",accuracy_score(y_dev,y_dev_pred))
print("NB Validation set precision:",precision_score(y_dev,y_dev_pred))
print("NB Validation set recall:",recall_score(y_dev,y_dev_pred))
print("NB Validation set f1_score:",f1_score(y_dev,y_dev_pred))

y_test_pred = NB.predict(X_test_Tfidf)
print("NB Test set accuracy:",accuracy_score(y_test,y_test_pred))
print("NB Test set precision:",precision_score(y_test,y_test_pred))
print("NB Test set recall:",recall_score(y_test,y_test_pred))
print("NB Test set f1_score:",f1_score(y_test,y_test_pred))

#-------------------------------------------------------------
# SVM model
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_train_Tfidf, y_train)
y_dev_pred = SVM.predict(X_dev_Tfidf)
print("SVM Validation set accuracy:",accuracy_score(y_dev,y_dev_pred))
print("SVM Validation set precision:",precision_score(y_dev,y_dev_pred))
print("SVM Validation set recall:",recall_score(y_dev,y_dev_pred))
print("SVM Validation set f1_score:",f1_score(y_dev,y_dev_pred))

y_test_pred = SVM.predict(X_test_Tfidf)
print("SVM Test set accuracy:",accuracy_score(y_test,y_test_pred))
print("SVM Test set precision:",precision_score(y_test,y_test_pred))
print("SVM Test set recall:",recall_score(y_test,y_test_pred))
print("SVM Test set f1_score:",f1_score(y_test,y_test_pred))

#--------------------------------------------------------------------
# Random forest model
RF = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
RF.fit(X_train_Tfidf, y_train)
y_dev_pred = RF.predict(X_dev_Tfidf)
print("Random Forest Validation set accuracy:",accuracy_score(y_dev,y_dev_pred))
print("Random Forest Validation set precision:",precision_score(y_dev,y_dev_pred))
print("Random Forest Validation set recall:",recall_score(y_dev,y_dev_pred))
print("Random Forest Validation set f1_score:",f1_score(y_dev,y_dev_pred))

y_test_pred = RF.predict(X_test_Tfidf)
print("Random Forest Test set accuracy:",accuracy_score(y_test,y_test_pred))
print("Random Forest Test set precision:",precision_score(y_test,y_test_pred))
print("Random Forest Test set recall:",recall_score(y_test,y_test_pred))
print("Random Forest Test set f1_score:",f1_score(y_test,y_test_pred))
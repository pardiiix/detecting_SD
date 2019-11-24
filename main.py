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
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Convolution1D, MaxPooling1D, Flatten, Conv1D
import matplotlib.pyplot as plt
import keras.metrics
from keras import backend as K
from keras import optimizers
from keras.layers import Dropout
from keras.layers.core import Reshape
# import autocorrect
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
from CNN_1D import create_cnn_model
from predict_model_confidence import predict_label



def main():
    create_cnn_model()
    print("model has been trained.\nThe model will now try to predict labels for other files.")
    predict_label('abdominal_comments.csv')

if __name__ == "__main__":
    main()
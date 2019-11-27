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
from train_self_active import create_semi_cnn_model, append_to_dataframe



def main():
    print("creating the first model by manually labeled data...")
    create_cnn_model()
    print("model has been trained.\nThe model will now try to predict labels for other files.")
    df = predict_label('testing_sample_active.csv') #predicts unlabbeled comments (this needs to be changed)
    # new_df = append_to_dataframe(df, testing_sample_active.csv_dataframe) #this needs to be changed
    new_df = df.to_csv("dataframe_testing_sample_active", mode='a', header=False) #creates csv file for unlabelled dataframe
    labeled_df = pd.read_pickle("sdnsd_labels_df.csv")
    print("new df:")
    print(new_df)

    print("labelled df:")
    print(labeled_df) #remove this
    import ipdb;ipdb.set_trace()

    create_semi_cnn_model(new_df) #this needs to be changed

if __name__ == "__main__":
    main()
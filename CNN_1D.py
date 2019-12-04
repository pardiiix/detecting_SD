import nltk
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer #is based on The Porter Stemming Algorithm
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
# from keras.layers.embeddings import Embedding
# from keras.layers.recurrent import LSTM, Bidirectional
# from keras.layers.core import Dense
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


def create_cnn_model():

    #reading the labelled comment file
    with open('SD', 'r') as file:
        sd = file.readlines()

    with open('no_SD', 'r') as file:
        no_sd = file.readlines()
        # lower_text = []
        # data = json.load(json_file)
        # for i in data["memory_loss"]:
        #     lower_text.append(to_lower(i["comments"])) #converting the comments in memory_loss dictionary to lower case

    df = pd.DataFrame(columns=['comments', 'exact_comments', 'polarity'])
    df['comments'] = no_sd + sd
    df['exact_comments'] = no_sd + sd
    df['polarity'] = [0] * len(no_sd) + [1] * len(sd)
    df = df.sample(frac=1, random_state = 10) #shuffling the rows
    df.reset_index(inplace = True, drop = True)
    df['comments'] = df['comments'].str.lower() #Converting text to lower case


    stopword = nltk.corpus.stopwords.words("english")
    not_stopwords = {'my', 'I', 'myself', 'me', 'i'} #removing some stopwords related to self-disclosure in nltk stopwords
    final_stop_words = set([word for word in stopword if word not in not_stopwords])
    speller = Speller()

    for i in range(len(df['comments'])):
        df['comments'][i] = re.sub("[0-9]+", " ", str(df['comments'][i])) #removing digits, since they're not important
        df['comments'][i] = deEmojify(df['comments'][i])
        df['comments'][i] = strip_punctuation(df['comments'][i])
        df['comments'][i] = ' '.join(speller(word) for word in df['comments'][i].split() if word not in final_stop_words) #removing stopwords and spell-correcting

    df.to_pickle('sdnsd_labels_df')

    max_sent_len = 200
    max_vocab_size = 4000
    word_seq = [text_to_word_sequence(comment) for comment in df['comments']]
    # print(word_seq)

    # vectorizing a text corpus, turning each text into either a sequence of integers (each integer being the index of a token in a dictionary)
    tokenizer = Tokenizer(num_words = max_vocab_size)
    tokenizer.fit_on_texts([' '.join(seq[:max_sent_len]) for seq in word_seq]) #Updates internal vocabulary based on a list of texts up to the max_sent_len.
    # print("vocab size: ", len(tokenizer.word_index)) #vocab size: 949

    #converting sequence of words to sequence of indices
    X = tokenizer.texts_to_sequences([' '.join(seq[:max_sent_len]) for seq in word_seq])
    X = pad_sequences(X, maxlen = max_sent_len, padding= 'post' , truncating='post')
    # X = np.expand_dims(X, axis =2) #reshape X to 3 dimensions
    # X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y = df['polarity']
    # y = np.expand_dims(y , axis =2)
    # print(X)

    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=10, test_size=0.2)
    X_train, X_dev, y_train, y_dev = train_test_split(X_test,y_test, random_state=10, test_size=0.3)

    #creating a dictionary for glove such that embeddings_dictionary[word] = word_vector
    embeddings_dictionary = dict()
    glove_file = open('glove.6B.100d.txt')
    for line in glove_file:
        records = line.split()
        word = records[0]
        word_vector = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = word_vector
    glove_file.close()


    # print(tokenizer.word_index)

    #creating an embedding matrix with words in our vocabulary and word vectors in glove
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    #building a sequential model by stacking neural net units
    model = Sequential()
    model.add(Embedding(input_dim = vocab_size,
                        output_dim = 100,
                        weights = [embedding_matrix],
                        input_length = max_sent_len,
                        trainable = False,
                        name = 'word_embedding_layer'
                        ))



    model.add(Conv1D(filters = 32, kernel_size = 8, activation='relu',  name = 'CNN_layer'))
    model.add(MaxPooling1D(10))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation= 'sigmoid', name = 'output_layer'))

    optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    # optimizer=keras.optimizers.SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer= optimizer, loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])

    # print(model.summary())
    # print("lr", K.eval(model.optimizer.lr))

    history = model.fit(X_train, y_train, batch_size=32, epochs=20, verbose = 1, validation_split =0.2) #verbose =1 : see trainig progress for each epoch


    loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose = 1)

    model.save("saved_cnn_model.h5")
    print("saved model to disk")

    # print(score)
    print("loss: ", loss)
    print("accuracy: ", accuracy)
    print("f1_score:", f1_score)
    print("precision:", precision)
    print("recall:", recall)


    # print(history.history)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper left')
    plt.show()




# x = re.findall("[^a-zA-Z]very$", ' '.join(c for c in df))
# print(x)

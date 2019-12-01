import pandas as pd
import re
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer #is based on The Porter Stemming Algorithm
from contractions import contractions_dict
from autocorrect import Speller
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

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

stopword = nltk.corpus.stopwords.words("english")
not_stopwords = {'my', 'I', 'myself', 'me'} #removing some stopwords related to self-disclosure in nltk stopwords
final_stop_words = set([word for word in stopword if word not in not_stopwords])
speller = Speller()

with open('depression_comments.csv', 'r') as file:
		dep_comments = file.readlines()

df = pd.DataFrame(columns=['comments', 'polarity'])
df['comments'] = dep_comments
df['comments'] = df['comments'].str.lower() #Converting text to lower case

for i in range(len(df['comments'])):
	df['comments'][i] = re.sub("[0-9]+", " ", str(df['comments'][i])) #removing digits, since they're not important
	df['comments'][i] = deEmojify(df['comments'][i])
	df['comments'][i] = strip_punctuation(df['comments'][i])
	df['comments'][i] = ' '.join(speller(word) for word in df['comments'][i].split() if word not in final_stop_words) #removing stopwords and spell-correcting
	if df['comments'][i] != 'reply':
		df['comments'][i] = df['comments'][i]
	else:
		df['comments'].drop(i, axis=0)
# for comment in df['comments']:
# 	if comment == reply or comment == report or comment < 4:
# 		pass
# 	else:
# 		df['comments'] = df['comments']
print(df['comments'])

max_sent_len = 80
max_vocab_size = 200
word_seq = [text_to_word_sequence(comment) for comment in df['comments']]
# print(word_seq)

# vectorizing a text corpus, turning each text into either a sequence of integers (each integer being the index of a token in a dictionary)
tokenizer = Tokenizer(num_words = max_vocab_size)
tokenizer.fit_on_texts([' '.join(seq[:max_sent_len]) for seq in word_seq]) #Updates internal vocabulary based on a list of texts up to the max_sent_len.


#converting sequence of words to sequence of indices
X = tokenizer.texts_to_sequences([' '.join(seq[:max_sent_len]) for seq in word_seq])
X = pad_sequences(X, maxlen = max_sent_len, padding= 'post' , truncating='post')

y = df['polarity']

print(X)
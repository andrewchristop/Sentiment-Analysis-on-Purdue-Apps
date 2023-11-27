import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import re
import string
import math
import nltk
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, optimizers, utils
from tensorflow.keras.preprocessing import sequence, text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer

#nltk.download("punkt")

def lemmatize_text(text):
  lem = WordNetLemmatizer()
  w_tokenizer = WhitespaceTokenizer()
  string = ""
  for w in w_tokenizer.tokenize(text):
    string = string + lem.lemmatize(w) + " "
  return string

data = pd.read_csv('/Users/christopherandrew/Documents/Sentiment-Analysis-on-Purdue-Apps/data/trainTest.csv')
stop = stopwords.words('english') + list(string.punctuation) 
data['Translated_Review'] = data['Translated_Review'].apply(lambda x: str(x).lower())
data['Translated_Review'] = data['Translated_Review'].str.replace('[^\w\s]','', regex=True)
data['Translated_Review'] = data['Translated_Review'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))
data['Translated_Review'] = data['Translated_Review'].apply(lambda x : ' '.join(x for x in x.split() if x.isdigit()==False))
data['Translated_Review'] = data.Translated_Review.apply(lemmatize_text)

reviews = data['Translated_Review']
labels = data['Sentiment']

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(reviews)
labels = labels.astype('category')

x_train, x_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

num_classes = 3
batch_size = 100
max_words = 400

x_train_binary = tokenizer.texts_to_matrix(x_train, mode='binary')
x_test_binary = tokenizer.texts_to_matrix(x_test, mode='binary')

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

y_train_ohe = utils.to_categorical(y_train, num_classes)
y_test_ohe = utils.to_categorical(y_test, num_classes)

embedding_dim = 128
vocab_size = 22000

model = models.Sequential()
x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)

x_train_seq = sequence.pad_sequences(x_train_seq, maxlen=max_words)
x_test_seq = sequence.pad_sequences(x_test_seq, maxlen=max_words)

model = models.Sequential()
model.add(layers.Embedding(vocab_size, 250, mask_zero=True))
model.add(layers.LSTM(128,dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
model.add(layers.LSTM(64,dropout=0.5, recurrent_dropout=0.5, return_sequences=False))
model.add(layers.Dense(num_classes,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=0.001),metrics=['accuracy'])
model.summary()

history = model.fit(x_train_seq, y_train_ohe, epochs=3, verbose=1)

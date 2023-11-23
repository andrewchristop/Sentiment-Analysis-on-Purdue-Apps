import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import re
import math
import nltk
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.preprocessing import sequence, text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer

def lemmatize_text(text):
  lem = WordNetLemmatizer()
  w_tokenizer = WhitespaceTokenizer()
  string = ""
  for w in w_tokenizer.tokenize(text):
    string = string + lem.lemmatize(w) + " "
  return string

data = pd.read_csv('/Users/christopherandrew/Documents/Sentiment-Analysis-on-Purdue-Apps/data/trainTest.csv')
stop = set(stopwords.words('english'))
data['Translated_Review'] = data['Translated_Review'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))
data['Translated_Review'] = data.Translated_Review.apply(lemmatize_text)

reviews = data['Translated_Review'].values
labels = data['Sentiment'].values
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
train_sentences, test_sentences, train_labels, test_labels = train_test_split(reviews, encoded_labels, stratify = encoded_labels)

max_review_length = 200
embedding_dim = 100
vocab_size = 20000
tokenizer = text.Tokenizer(oov_token="<LOV>") 
tokenizer.fit_on_texts(train_sentences)
tokenizer.fit_on_texts(test_sentences)

train_sequences = tokenizer.texts_to_sequences(train_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)

train_sentences = sequence.pad_sequences(train_sequences, maxlen=max_review_length)
test_sentences = sequence.pad_sequences(test_sequences, maxlen=max_review_length)

model = models.Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=max_review_length))
model.add(layers.Bidirectional(layers.LSTM(64)))
model.add(layers.Dense(24, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

adam = optimizers.Adam(learning_rate=0.001)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_sentences, train_labels, batch_size=32 , epochs=5, verbose=2, validation_split=0.1)
#print(train_sentences.shape)
#print(train_labels.shape)
#print(test_sentences.shape)
#print(test_labels.shape)
#print(model.summary())


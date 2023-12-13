#Importing necessary modules
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import re
import string
import math
import nltk
from pathlib import Path
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
from plot import *

#nltk.download("punkt")

#Function that lemmatizes text

def lemmatize_text(text):
  lem = WordNetLemmatizer()
  w_tokenizer = WhitespaceTokenizer()
  string = ""
  for w in w_tokenizer.tokenize(text):
    string = string + lem.lemmatize(w) + " "
  return string

#Loads training data 
data = pd.read_csv('/Users/christopherandrew/Documents/Sentiment-Analysis-on-Purdue-Apps/data/trainTest.csv')
data = data.drop(data[data['Sentiment'] == 'Neutral'].index) #Excludes reviews that are neutral in tone
stop = stopwords.words('english') + list(string.punctuation) #Defines stopwords to be excluded
data['Translated_Review'] = data['Translated_Review'].apply(lambda x: str(x).lower()) #Converts all reviews to lower case
data['Translated_Review'] = data['Translated_Review'].str.replace('[^\w\s]','', regex=True) #Removes punctuation from reviews
data['Translated_Review'] = data['Translated_Review'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)])) #Removes stopwords from the reviews
data['Translated_Review'] = data['Translated_Review'].apply(lambda x : ' '.join(x for x in x.split() if x.isdigit()==False)) #Removes numbers from the characters
data['Translated_Review'] = data.Translated_Review.apply(lemmatize_text) #Lemmatizes each word in the review

#Word cloud plotting of key words found in positive and negative reviews
positive = data[data['Sentiment'].str.contains('Positive')]
negative = data[data['Sentiment'].str.contains('Negative')]
word_cloud(positive['Translated_Review'].str.cat(sep=' '))
word_cloud(negative['Translated_Review'].str.cat(sep=' '))

reviews = data['Translated_Review']
labels = data['Sentiment']

#Tokenizing words
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(reviews)
labels = labels.astype('category')

#Train test split with 80% of the training data dedicated to training the model, and 20% of the data dedicated to testing the model
x_train, x_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

num_classes = 3
batch_size = 100
max_words = 400

#Further processing to resolve compatibility issues with the model
x_train_binary = tokenizer.texts_to_matrix(x_train, mode='binary')
x_test_binary = tokenizer.texts_to_matrix(x_test, mode='binary')

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

y_train_ohe = utils.to_categorical(y_train, num_classes)
y_test_ohe = utils.to_categorical(y_test, num_classes)

embedding_dim = 128
vocab_size = 21000

x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)

#Padding sequences to make sure that training and testing data is of uniform length
x_train_seq = sequence.pad_sequences(x_train_seq, maxlen=max_words)
x_test_seq = sequence.pad_sequences(x_test_seq, maxlen=max_words)

#Defines location of the saved model
path = Path('./saved_models/model.h5')

#Compiles model and saves it to designated directory if the model isn't already saved
if(path.is_file() == False):
  model = models.Sequential()
  model.add(layers.Embedding(vocab_size, 250, mask_zero=True))
  model.add(layers.LSTM(128,dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
  model.add(layers.LSTM(64,dropout=0.5, recurrent_dropout=0.5, return_sequences=False))
  model.add(layers.Dense(num_classes,activation='sigmoid'))
  model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=0.001),metrics=['accuracy'])
  model.summary()
  history = model.fit(x_train_seq, y_train_ohe, epochs=3, verbose=1)
  model.save('./saved_models/model.h5') 

#Loads the saved model
model = models.load_model('./saved_models/model.h5')

#Loading of the actual data gathered from the survey
sentence = pd.read_csv('/Users/christopherandrew/Documents/Sentiment-Analysis-on-Purdue-Apps/data/actualData.csv') 
app = sentence['app'].values
rev = sentence['review'].values
sentence_sequences = tokenizer.texts_to_sequences(rev)
padded_sentences = sequence.pad_sequences(sentence_sequences, maxlen=max_words)

# Make predictions using the loaded model
predictions = model.predict(padded_sentences)

predicted_labels = np.argmax(predictions, axis=1)

# Decode the predicted labels using the LabelEncoder
decoded_predicted_labels = le.inverse_transform(predicted_labels)

#Variables tied to positive and negative toned review count of each app which is useful for plotting
p_app_pos = 0
p_mob_pos = 0
p_ath_pos = 0
p_rec_pos = 0
p_gui_pos = 0

p_app_neg = 0
p_mob_neg = 0
p_ath_neg = 0
p_rec_neg = 0
p_gui_neg = 0

# Display the results
for i in range(len(sentence)):
    print(f"App: {app[i]}")
    print(f"Review: {rev[i]}")
    print(f"Predicted Sentiment: {decoded_predicted_labels[i]}")
    
    #Increments positive and negative variables for plotting
    if (decoded_predicted_labels[i] == "Positive"):
      if (app[i] == "Purdue Guide"):
        p_gui_pos = p_gui_pos + 1
      elif(app[i] == "Purdue App"):
        p_app_pos = p_app_pos + 1
      elif(app[i] == "Purdue Athletics"):
        p_ath_pos = p_ath_pos + 1
      elif(app[i] == "Purdue RecWell"):
        p_rec_pos = p_rec_pos + 1
      else:
        p_mob_pos = p_mob_pos + 1
    else:
      if (app[i] == "Purdue Guide"):
        p_gui_neg = p_gui_neg + 1
      elif(app[i] == "Purdue App"):
        p_app_neg = p_app_neg + 1
      elif(app[i] == "Purdue Athletics"):
        p_ath_neg = p_ath_neg + 1
      elif(app[i] == "Purdue RecWell"):
        p_rec_neg = p_rec_neg + 1
      else:
        p_mob_neg = p_mob_neg + 1
    print()

#Plotting the positive and negative reviews of each app
bar_pos(p_gui_pos, p_app_pos, p_ath_pos, p_rec_pos, p_mob_pos)
bar_neg(p_gui_neg, p_app_neg, p_ath_neg, p_rec_neg, p_mob_neg)

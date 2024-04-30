import pandas as pd
import tensorflow as tf

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import pickle
import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from word_process import WordProcess

nltk.download('stopwords')


Dense = tf.keras.layers.Dense
Embedding = tf.keras.layers.Embedding
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
Sequential = tf.keras.models.Sequential
one_hot = tf.keras.preprocessing.text.one_hot
LSTM = tf.keras.layers.LSTM
Bidirectional = tf.keras.layers.Bidirectional
Dropout = tf.keras.layers.Dropout
Tokenizer =  tf.keras.preprocessing.text 
EarlyStopping = tf.keras.callbacks

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow import Tokenizer 



df = pd.read_csv("C:\\Users\\aller\\Desktop\\chatbot5\\datasets\\final_dataset.csv")

# print(df.head())

df = df.dropna()

# print(df.columns)

voc_size=50000

df.reset_index(inplace=True)


### Dataset Preprocessing
from word_process import WordProcess

wp = WordProcess()

corpus = []

for i in range(0, len(df)):
    review = wp.process_sent2sent(df['Symptoms'][i])
    corpus.append(review)

# print(corpus[:5])


# Initialize Tokenizer
tokenizer = Tokenizer(lower=True, split=' ', oov_token='<OOV>')
tokenizer.fit_on_texts(corpus)

# Convert texts to sequences of integers
onehot_repr = tokenizer.texts_to_sequences(corpus)

with open(f'C:\\Users\\aller\\Desktop\\chatbot5\\models\\tokenizer_cnn_v1.pkl', 'wb') as f:  # Open the file in write-binary mode
    pickle.dump(tokenizer, f) 

# making all the sentences of the same length
sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

print(embedded_docs)

embedding_vector_features=40

num_classes = 30
model = Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_shape=(sent_length,)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

df['Disease'] =  df['Disease'].apply(lambda x: x.lower())
diseases = pd.get_dummies(df['Disease']).columns

print(diseases.get_loc('common cold'), df[df['Disease'] == 'common cold'].iloc[0]['Symptoms'])

y = df['Disease'].apply(lambda x: diseases.get_loc(x))


# saving titles of diseases to a text file for future use

with open(f'../models/disease_classes_{version}.txt', 'w') as f:
    f.write(json.dumps( pd.get_dummies(df['Disease']).columns.values.tolist()))



X_final=np.array(embedded_docs)
y_final=np.array(y)

X_final.shape,y_final.shape

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)


# early_stopping means that the model will stop training if the validation loss does not decrease for 3 epochs
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=3,          # Wait for 5 epochs without improvement 
    mode='min',          # Minimize validation loss
    restore_best_weights=True  # Restore weights from the best epoch
)

##Training
history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=15,batch_size=64,callbacks=[early_stopping])

# plot line graph of history of accuracy and val_accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

y_pred=model.predict(X_test)

y_pred_t = [i.argmax() for i in y_pred]

accuracy_score(y_test,y_pred_t)

model.save_weights(f'../models/model_v1.weights.h5')
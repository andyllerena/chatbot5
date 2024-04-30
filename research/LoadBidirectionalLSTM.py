import tensorflow as tf
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

# from keras.layers import Conv1D, MaxPooling1D 

Conv1D = tf.keras.layers.Conv1D
MaxPooling1D = tf.keras.layers.MaxPooling1D


import nltk
import re
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split

# version = "cnn_v1"

model_path = f"C:\\Users\\aller\\Desktop\\chatbot5\\models\\model_cnn_v1.weights.h5"
tokenizer_path = f"C:\\Users\\aller\\Desktop\\chatbot5\\models\\tokenizer_cnn_v1.pkl"
class_path = f"C:\\Users\\aller\\Desktop\\chatbot5\\models\\disease_classes_cnn_v1.txt"

## Creating the model_loaded based on training model_loaded architecture
sent_length=20
voc_size = 50000
embedding_vector_features=40
model_loaded = None
num_classes = 30

# # Define CNN-LSTM model_loaded
model_loaded = Sequential()
model_loaded.add(Embedding(voc_size,embedding_vector_features,input_shape=(sent_length,)))
model_loaded.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model_loaded.add(MaxPooling1D(pool_size=2))
model_loaded.add(LSTM(100))
model_loaded.add(Dropout(0.3))
model_loaded.add(Dense(num_classes,activation='softmax'))
model_loaded.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


print(model_loaded.summary())

# loading saved model weights
model_loaded.load_weights(model_path)

# Preprocessing function
import nltk
import re
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from word_process import WordProcess

nltk.download('stopwords')

tokenizer = pickle.load(open(tokenizer_path, 'rb'))
stop_words = stopwords.words('english')
stemmer = PorterStemmer()
wp = WordProcess()



def preprocess_text(text):
    """Preprocesses a single text sample for disease prediction."""
    # voc_size = 5000
    sent_length = 20
    processed_text = wp.process_sent2sent(text)

    # One-hot encoding and padding
    # print(processed_text)
    onehot_vector = tokenizer.texts_to_sequences([processed_text])
    # print('vector',onehot_vector)
    padded_vector = pad_sequences(onehot_vector, padding='pre', maxlen=sent_length)

    return padded_vector[0].tolist()

with open(class_path) as f:
    disease_classes = json.load(f)
disease_classes, len(disease_classes)

test_cases = [
"I have been sneezing frequently, accompanied by a mild headache, runny nose, and a general feeling of being unwell.",
"Experiencing a low-grade fever with chills, nasal congestion, and a scratchy throat.",
"Mild body aches with a runny nose, a few sneezes, and feeling slightly fatigued.",
"Congested nose with a sore throat, slight cough, and sneezing fits.",
"I am experiencing itching and irritation in the vaginal area, along with a white, clumpy discharge that resembles cottage cheese.",
"There's a burning sensation during urination and redness and swelling of the vulva.",
"Feeling soreness and experiencing painful intercourse, accompanied by a thick, odorless, white vaginal discharge.",
"Persistent itching and a thick white discharge, with slight redness around the external genitalia.",
"Feeling tired all the time and my bones ache, especially in the joints and back. There's also muscle weakness.",
"Noticing more hair falling out, general fatigue, and aching bones. I've been indoors most of the time.",
"Experiencing bone pain and muscle weakness, feeling depressed more frequently.",
"My doctor mentioned bone softening, and I feel persistently low energy and down in mood.",
"My stomach cramps after eating and I frequently have diarrhea or constipation, feeling bloated.",
"Experiencing abdominal pain, bloating, and an inconsistent stool pattern, swinging between diarrhea and constipation.",
"Frequent bloating and gas with episodes of constipation followed by sudden diarrhea.",
"Abdominal discomfort, altered bowel habits, with bouts of diarrhea and periods of constipation, including bloating."
]

for test in test_cases:
    print(test)
    ind = model_loaded.predict(np.array([preprocess_text(test)]),verbose=0).argmax()
    print( disease_classes[ind])
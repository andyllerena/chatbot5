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

import nltk
import re
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split

model_path = "C:\\Users\\aller\\Desktop\\chatbot5\\models\\biLstm_v2\\model.weights.h5"
tokenizer_path = "C:\\Users\\aller\\Desktop\\chatbot5\\models\\biLstm_v2\\tokenizer.pkl"

## Creating the model based on training model architecture
sent_length=20
voc_size = 5000
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_shape=(sent_length,)))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.3))
model.add(Dense(773,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# print(model.summary())

model.load_weights(model_path)

tokenizer = pickle.load(open(tokenizer_path, 'rb'))
stop_words = stopwords.words('english')
stemmer = PorterStemmer()

with open("C:\\Users\\aller\\Desktop\\chatbot5\\models\\biLstm_v2\\disease_classes.json") as f:
    disease_classes = json.load(f)
print(type(disease_classes))  # Check if it's a dictionary



def preprocess_text(text):
    """Preprocesses a single text sample for disease prediction."""
    # voc_size = 5000
    sent_length = 20
    # Cleaning
    text = re.sub('[^a-zA-Z]', ' ', text) 
    text = text.lower()

    # Tokenization
    words = text.split()

    # Stop word removal and stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]

    # Recombine text
    processed_text = ' '.join(words)

    # One-hot encoding and padding
    # print(processed_text)
    onehot_vector = tokenizer.texts_to_sequences([processed_text])
    # print('vector',onehot_vector)
    padded_vector = pad_sequences(onehot_vector, padding='pre', maxlen=sent_length)

    return padded_vector[0].tolist()

# Test cases
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


# Processing each test case and making predictions
for case in test_cases:
    processed_input = preprocess_text(case)
    prediction = model.predict(np.array([processed_input]))
    predicted_index = prediction.argmax()
    predicted_disease = disease_classes[predicted_index]
    print(f"Symptoms: {case}\nPredicted Disease: {predicted_disease}\n")
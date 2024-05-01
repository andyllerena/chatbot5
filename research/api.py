
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow import keras
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPooling1D, Bidirectional, Dropout
from keras.preprocessing.sequence import pad_sequences
import os
from word_process import WordProcess
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import json
import numpy as np
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = ""

app = Flask(__name__)
CORS(app)

wp = WordProcess()

model_path = f"C:\\Users\\aller\\Desktop\\chatbot5\\models\\model_cnn_v1.weights.h5"
tokenizer_path = f"C:\\Users\\aller\\Desktop\\chatbot5\\models\\tokenizer_cnn_v1.pkl"
class_path = f"C:\\Users\\aller\\Desktop\\chatbot5\\models\\disease_classes_cnn_v1.txt"

sent_length=20
voc_size = 50000
embedding_vector_features=40
model_loaded = None
num_classes = 30

model_loaded = Sequential()
model_loaded.add(Embedding(voc_size,embedding_vector_features,input_shape=(sent_length,)))
model_loaded.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model_loaded.add(MaxPooling1D(pool_size=2))
model_loaded.add(LSTM(100))
model_loaded.add(Dropout(0.3))
model_loaded.add(Dense(num_classes,activation='softmax'))
model_loaded.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model_loaded.load_weights(model_path)


nltk.download('stopwords')

tokenizer = pickle.load(open(tokenizer_path, 'rb'))
stop_words = stopwords.words('english')
stemmer = PorterStemmer()

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

@app.route('/chatbot', methods=['POST'])
def detect_disease():
    user_input = request.json['symptoms']
    ind = model_loaded.predict(np.array([preprocess_text(user_input)]), verbose=0).argmax()

    return jsonify({"disease": disease_classes[ind]}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

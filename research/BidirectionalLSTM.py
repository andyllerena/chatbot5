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

# from tensorflow.keras.preprocessing.text import Tokenizer




df = pd.read_csv("C:\\Users\\aller\\Desktop\\chatbot5\\datasets\\symptoms_diseases_mod_v1.csv")

df = df.dropna()

print(df.columns)

voc_size = 5000

nltk.download("stopwords")

ps = PorterStemmer()
corpus = []
for i in range(0, len(df)):
    # print(i)
    review = re.sub("[^a-zA-Z]", " ", df["symptoms_processed"][i])
    review = review.lower()
    review = review.split()

    review = [
        ps.stem(word) for word in review if not word in stopwords.words("english")
    ]
    review = " ".join(review)
    corpus.append(review)

# Initialize Tokenizer
Tokenizer = Tokenizer()
Tokenizer.fit_on_texts(corpus)

# Convert texts to sequences of integers
onehot_repr = Tokenizer.texts_to_sequences(corpus)

# saving the tokenizer for future use
# have commented this out as it is not needed for now
with open('tokenizer.pkl', 'wb') as f:  # Open the file in write-binary mode
    pickle.dump(Tokenizer, f)  

   

#Embedding Representation
sent_length = 20
embedded_docs = pad_sequences(onehot_repr, padding="pre", maxlen=sent_length)

print(embedded_docs[0])

## Creating model
embedding_vector_features=40
model1=Sequential()
model1.add(Embedding(voc_size,embedding_vector_features,input_shape=(sent_length,)))
model1.add(Bidirectional(LSTM(100)))
model1.add(Dropout(0.3))
model1.add(Dense(773,activation='sigmoid'))
model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model1.summary())

y = pd.get_dummies(df['disease'])

# saving titles of diseases to a text file for future use
with open('disease_classes.txt', 'w') as f:
    f.write(json.dumps(y.columns.values.tolist()))

print(len(embedded_docs),y.shape)

X_final=np.array(embedded_docs)
y_final=np.array(y)

print(X_final.shape,y_final.shape)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=3,          # Wait for 5 epochs without improvement 
    mode='min',          # Minimize validation loss
    restore_best_weights=True  # Restore weights from the best epoch
)

### Finally Training
model1.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=30,batch_size=64,callbacks=[early_stopping])

y_pred=model1.predict(X_test)

y_test_t = [i.argmax() for i in y_test]
y_pred_t = [i.argmax() for i in y_pred]

print(accuracy_score(y_test_t,y_pred_t))

# save model weights
model1.save_weights('model.weights.h5')

# model definition be initialize with this code
embedding_vector_features=40
model_loaded=Sequential()
model_loaded.add(Embedding(voc_size,embedding_vector_features,input_shape=(sent_length,)))
model_loaded.add(Bidirectional(LSTM(100)))
model_loaded.add(Dropout(0.3))
model_loaded.add(Dense(773,activation='sigmoid'))
model_loaded.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model_loaded.summary())

# finally load the weights from the saved model
model_loaded.load_weights('model.weights.h5')

y_pred_ml=model_loaded.predict(X_test)
y_pred_t = [i.argmax() for i in y_pred_ml]

print(accuracy_score(y_test_t,y_pred_t))
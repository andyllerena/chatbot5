# Disease Prediction Chatbot NLP Model

## Overview
This project is a disease prediction chatbot that uses a Recurrent neural network (RNN) and long short-term memory (LSTM) model to predict diseases based on symptoms provided by the user. The chatbot is designed to assist users in understanding their medical symptoms and provide information on potential diseases.

## Project Structure
- `datasets/`: Contains the dataset used for training the model.
- `models/`: Contains the saved model weights and tokenizer for inference.
- `word_process.py`: Python file containing the WordProcess class for text preprocessing.
- `BidirectionalLSTM.py`: Python script used for training the disease prediction model.
- `LoadBidirectionalLSTM.py`: Python script for making predictions using the trained model.
- `requirements.txt`: Contains the necessary Python packages for running the project.

## Installation
1. Clone the repository:
   ```bash
   https://github.com/andyllerena/chatbot5.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   

## Sample Input
```
"I have been sneezing frequently, accompanied by a mild headache, runny nose, and a general feeling of being unwell."
```

## Sample Output
```
"Common cold"
```

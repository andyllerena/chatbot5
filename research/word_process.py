import pandas as pd
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize


class WordProcess:
    def __init__(self):
        self.lemm = WordNetLemmatizer()
        nltk.download("stopwords")
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')

    def get_wordnet_pos(self, pos):
        if pos.startswith('J'):
            return wordnet.ADJ
        elif pos.startswith('V'):
            return wordnet.VERB
        elif pos.startswith('N'):
            return wordnet.NOUN
        elif pos.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN
        
    def process_sentence(self, sent):
        
        # lowercasing
        sen = sent.lower()

        # tokenizing
        tkns = wordpunct_tokenize(sen)

        # removing stopwords and punctuations
        stops = stopwords.words('english')
        stops.extend(["..","...",])
        puncts = string.punctuation
        clean = []
        for word in tkns:
            if len(word) > 1 and word not in stops and word not in puncts:
                clean.append(word)

        # word lemmatization
        word_tags = nltk.pos_tag(clean)
        word_lemm = []
        for word,tag in word_tags:
            word_lemm.append(self.lemm.lemmatize(word,self.get_wordnet_pos(tag)))

        sen = None
        tkns = None
        clean = None
        word_tags = None

        return word_lemm
    
    def process_sent2sent(self, sent):
        return " ".join(self.process_sentence(sent))
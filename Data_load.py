import re
import os
from keras.datasets import imdb
import pandas
from nltk.corpus import stopwords, reuters
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import sys
import numpy as np
import os
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re

cachedStopWords = stopwords.words("english")

GLOVE_DIR = "D:/glove/"
def tokenize(text):
  min_length = 3
  words = map(lambda word: word.lower(), word_tokenize(text))
  words = [word for word in words if word not in cachedStopWords]
  tokens = (list(map(lambda token: PorterStemmer().stem(token),
                                   words)))
  p = re.compile('[a-zA-Z]+');
  filtered_tokens = list(filter (lambda token: p.match(token) and
                               len(token) >= min_length,
                               tokens))
  return filtered_tokens


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    text = string
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    string = text.lower()

    return string.strip().lower()

def text_cleaner(text):
    text = text.replace(".", "")
    text = text.replace("[", " ")
    text = text.replace(",", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("\"", "")
    text = text.replace("-", " ")
    text = text.replace("=", " ")
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    return text.lower()


def loadData_Tokenizer(X_train, X_test,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH):
    np.random.seed(7)
    text = np.concatenate((X_train, X_test),axis=0)
    text = np.array(text)
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Found %s unique tokens.' % len(word_index))
    indices = np.arange(text.shape[0])
    #np.random.shuffle(indices)
    text = text[indices]
    print(text.shape)
    X_train = text[0:len(X_train),]
    X_test = text[len(X_train):,]
    GLOVE_DIR = "D:/glove/"
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            print("Warnning"+str(values)+" in" + str(line))
        embeddings_index[word] = coefs
    f.close()
    print('Total %s word vectors.' % len(embeddings_index))
    return (X_train, X_test, word_index,embeddings_index)


def Load_data(MAX_NB_WORDS,MAX_SEQUENCE_LENGTH):

    import pandas as pd
    content = pd.read_csv("D:\CHI\X_new.csv", encoding="utf-8")
    Label = pd.read_csv("D:\CHI\Y.csv", encoding="utf-8")
    content = content.as_matrix()
    content = np.array(content).ravel()
    Label = Label.as_matrix()
    Label = np.matrix(Label)
    np.random.seed(7)
    # print(Label)
    X, X_t, y_train, y_test = train_test_split(content, Label, test_size=0.3, random_state=1)
    X_train, X_test = loadData(X, X_t)
    X_train_M, X_test_M, word_index, embeddings_index = loadData_Tokenizer(X, X_t, MAX_NB_WORDS,
                                                                           MAX_SEQUENCE_LENGTH)
    number_of_classes = np.max(y_train)+1
    return (X_train,X_train_M, y_train,X_test, X_test_M, y_test, word_index, embeddings_index, number_of_classes)


def loadData(X_train, X_test):
    stop_words = stopwords.words("english")
    vectorizer_x = TfidfVectorizer()
    #vectorizer_x = CountVectorizer()
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    print(np.array(X_train).shape)

    return (X_train,X_test)
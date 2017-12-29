import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="2,1,0"
from sklearn.metrics import accuracy_score
from keras.datasets import cifar,mnist,imdb
import numpy as np
import itertools
import scipy.io as sio
import matplotlib.pyplot as plt
import gc
from operator import itemgetter
from keras.datasets import cifar10,cifar100
from sklearn.metrics import confusion_matrix
import random
import collections
from keras.models import Sequential
import Data_load
from sklearn.metrics import f1_score,precision_recall_fscore_support
import BuildModel
from sklearn.feature_extraction.text import CountVectorizer
from itertools import chain
from keras.callbacks import ModelCheckpoint
np.random.seed(7)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



def getUniqueWords(allWords) :
    uniqueWords = []
    for i in allWords:
        if not i in uniqueWords:
            uniqueWords.append(i)
    return uniqueWords
def column(matrix,i):
    f = itemgetter(i)
    return map(f,matrix)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])




def FilterByCluster(input_x,input_y,target):
    out = [row for row in input_x if target in input_y[row]]
    return (out)

def keyword_indexing(contentKey):
    vocabulary = list(map(lambda x: x.split(';'), contentKey))
    vocabulary = list(np.unique(list(chain(*vocabulary))))

    vec = CountVectorizer(vocabulary=vocabulary, tokenizer=lambda x: x.split(';'))
    out = np.array(vec.fit_transform(contentKey).toarray())
    print(out.shape)





if __name__ == "__main__":

    MEMORY_MB_MAX = 1600000
    MAX_SEQUENCE_LENGTH = 750
    MAX_NB_WORDS = 75000
    EMBEDDING_DIM = 100
    batch_size = 128
    n_epochs = 10
    sparse_categorical=0
    text=1
    np.set_printoptions(threshold=np.inf)
    np.random.seed(7)
    if text==1:
        y_proba = []
        model_DNN = []
        model_RNN = []
        model_CNN = []
        History = []
        score = []
        X_train,X_train_M, y_train,X_test, X_test_M, y_test, word_index, embeddings_index, number_of_classes = Data_load.Load_data(MAX_NB_WORDS,MAX_SEQUENCE_LENGTH)

        print("DNN ")
        filepath = "weights_DNN.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]

        model_DNN, model_tmp = BuildModel.buildModel_DNN_Tex(X_train.shape[1],number_of_classes,sparse_categorical)
        h = model_DNN.fit(X_train, y_train,
                     validation_data=(X_test, y_test),
                     epochs=n_epochs,
                     batch_size=batch_size,
                     callbacks=callbacks_list,
                     verbose=2)
        History.append(h)

        model_tmp.load_weights("weights_DNN.hdf5")
        if sparse_categorical==0:
            model_tmp.compile(loss='sparse_categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            y_pr = model_tmp.predict_classes(X_test, batch_size=batch_size)
            y_proba.append(np.array(y_pr))
            score.append(accuracy_score(y_test, y_pr))
        else:
            model_tmp.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            y_pr = model_tmp.predict(X_test, batch_size=batch_size)
            y_pr = np.argmax(y_pr,axis=1)
            y_proba.append(np.array(y_pr))
            y_test_temp = np.argmax(y_test,axis=1)
            score.append(accuracy_score(y_test_temp, y_pr))
        del model_tmp
        del model_DNN
        gc.collect()

    print(score)


import random
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.utils.training_utils import multi_gpu_model
from keras.models import Sequential
import keras
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import numpy as np
from keras.constraints import maxnorm
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D,Conv2D,MaxPooling2D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional,TimeDistributed,Convolution2D,Activation,GlobalAveragePooling2D,Convolution3D,GlobalAveragePooling3D
np.random.seed(7)
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.merge import Concatenate
import tensorflow as tf


def slice_batch(x, n_gpus, part):
    """
    Divide the input batch into [n_gpus] slices, and obtain slice number [part].
    i.e. if len(x)=10, then slice_batch(x, 2, 1) will return x[5:].
    """
    sh = K.shape(x)
    L = sh[0] // n_gpus
    if part == n_gpus - 1:
        return x[part*L:]
    return x[part*L:(part+1)*L]

def to_multi_gpu(model, n_gpus=2):
    """
    Given a keras [model], return an equivalent model which parallelizes
    the computation over [n_gpus] GPUs.

    Each GPU gets a slice of the input batch, applies the model on that slice
    and later the outputs of the models are concatenated to a single tensor,
    hence the user sees a model that behaves the same as the original.
    """
    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:], name="input1")

    towers = []
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_g = Lambda(slice_batch,
                             lambda shape: shape,
                             arguments={'n_gpus':n_gpus, 'part':g})(x)
            towers.append(model(slice_g))

    with tf.device('/cpu:0'):
        merged = Concatenate(axis=0)(towers)

    return Model(inputs=[x], outputs=[merged])



def buildModel_DNN_Tex(shape, nClasses,sparse_categorical):
    model = Sequential()
    Numberof_NOde =  1000
    nLayers = 4
    Numberof_NOde_old = Numberof_NOde
    model.add(Dense(Numberof_NOde,input_dim=shape,activation='relu'))
    model.add(Dropout(0.75))
    for i in range(0,nLayers):
        model.add(Dense(Numberof_NOde,input_dim=Numberof_NOde_old,activation='relu'))
        model.add(Dropout(0.75))
        Numberof_NOde_old = Numberof_NOde
    model.add(Dense(nClasses, activation='softmax'))
    model_tem = model
    if sparse_categorical==0:
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    return model,model_tem


def buildModel_RNN(word_index, embeddings_index, nClasses, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,sparse_categorical):
    model = Sequential()
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))

    G_L = 64
    print(G_L)
    for i in range(0,3):
        model.add(GRU(G_L,return_sequences=True, recurrent_dropout=0.2))
        model.add(Dropout(0.25))
    model.add(GRU(G_L, recurrent_dropout=0.2))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nClasses, activation='softmax'))

    model_tmp = model


    if sparse_categorical==0:
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    return model,model_tmp


def buildModel_CNN(word_index,embeddings_index,nClasses,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,sparse_categorical=0):
    model = Sequential()
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    # applying a more complex convolutional approach
    convs = []
    filter_sizes = []
    layer = 4
    print("Filter  ",layer)
    for fl in range(0,layer):
        filter_sizes.append((fl+2))

    node = 128
    print("Node  ", node)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    for fsz in filter_sizes:
        l_conv = Conv1D(node, kernel_size=fsz, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(5)(l_conv)
        #l_pool = Dropout(0.25)(l_pool)
        convs.append(l_pool)

    l_merge = Merge(mode='concat', concat_axis=1)(convs)
    l_cov1 = Conv1D(node, 5, activation='relu')(l_merge)
    l_cov1 = Dropout(0.25)(l_cov1)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(node, 5, activation='relu')(l_pool1)
    l_cov2 = Dropout(0.25)(l_cov2)
    l_pool2 = MaxPooling1D(30)(l_cov2)
    l_flat = Flatten()(l_pool2)
    values = list(range(250,1000))
    node = random.choice(values)
    l_dense = Dense(node, activation='relu')(l_flat)
    l_dense = Dropout(0.5)(l_dense)
    preds = Dense(nClasses, activation='softmax')(l_dense)

    model = Model(sequence_input, preds)
    model_tmp = model
    if sparse_categorical == 0:
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])


    return model,model_tmp

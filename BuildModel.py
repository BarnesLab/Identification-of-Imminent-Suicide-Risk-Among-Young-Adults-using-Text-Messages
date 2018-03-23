import random
from keras.utils.training_utils import multi_gpu_model
from keras.models import Sequential
import keras
import numpy as np
from keras.constraints import maxnorm
from keras.layers import Dense, Input, Flatten,Dropout

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


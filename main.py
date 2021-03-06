import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="2,1,0"
from sklearn.metrics import accuracy_score
import numpy as np
import itertools
import matplotlib.pyplot as plt
import gc
import Data_load
import BuildModel
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








if __name__ == "__main__":

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
        X_train, y_train,X_test, y_test, number_of_classes = Data_load.Load_data()

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


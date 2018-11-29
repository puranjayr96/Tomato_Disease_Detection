from __future__ import print_function
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.layers import Input, AveragePooling2D, Dropout
from sklearn.model_selection import train_test_split
from keras.models import Model
from sklearn.metrics import f1_score
from keras.layers.core import Dense, Activation, Flatten, Dropout
import math
from keras import optimizers
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
import codecs
import sklearn.metrics as sklm
from keras.utils.np_utils import to_categorical
import keras
import h5py
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import keras.backend as K
import sys
import json

import os
ls1=os.listdir('color')
if '.DS_Store' in ls1:
    ls1.remove('.DS_Store')
dic1={}
for idx,i in enumerate(ls1):
     dic1[i]=idx
import scipy.misc as sm
import numpy as np
# count=0
# for idx,i in enumerate(ls1):
#     dic1[i]=idx
#     ls2=os.listdir('color/'+i)
#     if '.DS_Store' in ls2:
#         ls2.remove('.DS_Store')
#     for j in ls2:
#         #im1=np.asarray(sm.imread('color/'+i+'/'+j))
#         #temp=np.zeros((len(im1),len(im1[0]),len(im1[0][0])   ))
#         count=count+1
# print("Reach 1 \n")
# ls1=os.listdir('color')
# if '.DS_Store' in ls1:
#     ls1.remove('.DS_Store')
# dic1={}
# X=np.zeros((count,256,256,3))
# Y=np.zeros((count,1))
# vap=0
#
# for idx,i in enumerate(ls1):
#     dic1[i]=idx
#     ls2=os.listdir('color/'+i)
#     if '.DS_Store' in ls2:
#         ls2.remove('.DS_Store')
#     for j in ls2:
#         im1=np.asarray(sm.imread('color/'+i+'/'+j))
#         print(str(im1.shape)+ " "+ i + " " + j)
#         X[vap,:,:,:]=im1
#         Y[vap,0]=idx
#         #temp=np.zeros((len(im1),len(im1[0]),len(im1[0][0])   ))
#         vap=vap+1
# def f1_mean_test(y_true, y_pred):
#     with tf.Session() as sess1:
#         K.set_session(sess1)
#         K.set_learning_phase(1)
#         def recall(y_true, y_pred):
#             """Recall metric.
#
#             Only computes a batch-wise average of recall.
#
#             Computes the recall, a metric for multi-label classification of
#             how many relevant items are selected.
#             """
#             true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#             print("true positive")
#
#             possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#             print("possible_positives")
#             recall = true_positives / (possible_positives + K.epsilon())
#             print("recall")
#             return recall
#
#         def precision(y_true, y_pred):
#             """Precision metric.
#
#             Only computes a batch-wise average of precision.
#
#             Computes the precision, a metric for multi-label classification of
#             how many selected items are relevant.
#             """
#             true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#             print("true positive")
#             predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#             print("predicted_positives")
#             precision = true_positives / (predicted_positives + K.epsilon())
#             print("precision")
#             return precision
#         precision = precision(y_true, y_pred)
#         recall = recall(y_true, y_pred)
#         print("f1")
#         return 2*((precision*recall)/(precision+recall+K.epsilon()))

#F1 through callback
class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.confusion = []
        self.precision = []
        self.recall = []
        self.f1s = []

    def on_epoch_end(self, epoch, logs={}):
        score = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
        targ = self.validation_data[1]

        self.f1s.append(sklm.f1_score(targ, predict,average='micro'))
        self.confusion.append(sklm.confusion_matrix(targ.argmax(axis=1),predict.argmax(axis=1)))

        return


h5f = h5py.File('variables.h5','r')
X = h5f['X'][:]
print("Reach 1")
Y = h5f['Y'][:]
print("Reach 2 \n")

batch_size = 64

num_classes = len(dic1)
epochs = 30

# input image dimensions
img_rows, img_cols = 256, 256
h = 299
w = 299
ch = 3
print("Reach 2.5 \n")
#tensor. will receive cifar10 images as input, gets passed to resize_images
img_placeholder = tf.placeholder("uint8", (None, 256, 256, 3))

#tensor. resized images. gets passed into Session()
resize_op = tf.image.resize_images(img_placeholder, (h, w), method=0)


# create a generator for batch processing
# this gen is written as if you could run through ALL of the data
# AWS instance doesn't have enough memory to hold the entire training bottleneck in memory
# so we will call for 10000 samples when we call it
def gen(session, data, labels, batch_size):
    def _f():
        start = 0
        end = start + batch_size
        n = data.shape[0]
        max_iter = math.ceil(n/batch_size)
        while True:
            # run takes in a tensor/function and performs it.
            # almost always, that function will take a Tensor as input
            # when run is called, it takes a feed_dict param which translates
            # Tensors into actual data/integers/floats/etc
            # this is so you can write a network and only have to change the
            # data being passed in one place instead of everywhere
            for i in range(0,max_iter):
                # X_batch is resized
                X_batch = session.run(resize_op, {img_placeholder: data[start:end]})
                # X_batch is normalized
                X_batch = preprocess_input(X_batch)
                y_batch = labels[start:end]
                start += batch_size
                end += batch_size
                if start >= n:
                    # start = 0
                    # end = batch_size
                    print("Bottleneck predictions completed.")
                    # break

                yield (X_batch, y_batch)

    return _f


def create_model_inception_resnet_v2():
    input_tensor = Input(shape=(h, w, ch))
    model = InceptionResNetV2(input_tensor=input_tensor, include_top=False)
    # x = model.output
    # x = Dropout(0.6)(x)
    # x = AveragePooling2D((7, 7))(x)
    # x = Dropout(0.6)(x)
    # model = Model(model.input, x)
    return model
print("Reach 2.8 \n")
X_train1, X_val1, y_train1, y_val1 = train_test_split(X, Y, test_size=0.3, random_state=0,shuffle=True)
print("Reach 3 \n")
print(X_train1.shape)
print(y_train1.shape)
with tf.Session() as sess:
    K.set_session(sess)
    K.set_learning_phase(1)

    model = create_model_inception_resnet_v2()

    train_gen = gen(sess, X_train1, y_train1, 64)
    print("Did gen")
    bottleneck_features_train = model.predict_generator(train_gen(), 2000)
    print("conv to train list")
    # bottleneck_features_train_list = bottleneck_features_train.tolist()
    print("conv to train list complete")
    print(bottleneck_features_train.shape)
    bftx = h5py.File('bftx_inception_resnet_v2.h5','w')
    bftx.create_dataset('bftx',data = bottleneck_features_train)
    bftx.close()
    with open('bfty_inception_resnet_v2.pkl', 'wb') as f:
        pickle.dump(y_train1, f)
    # datax = {'features': bottleneck_features_train, 'labels': y_train1}
    filepathx = 'resnet_train_bottleneck.json'
    # save_as_pickled_object(data, filepathx)
    # json.dump(data, codecs.open('resnet_train_bottleneck.json', 'w',encoding='utf-8'),separators=(',', ':'))
    print("json train dump complete")
    # h5y = h5py.File('resnet_train_bottleneck.h5', 'w')
    # h5y.create_dataset('data',data = data)

    val_gen = gen(sess, X_val1, y_val1, batch_size)
    bottleneck_features_validation = model.predict_generator(val_gen(), 2000)
    print("conv to val list")
    # bottleneck_features_validation_list = bottleneck_features_validation.tolist()
    print("conv to val list complete")
    bfvx = h5py.File('bfvx_inception_resnet_v2.h5','w')
    bfvx.create_dataset('bfvx',data = bottleneck_features_validation)
    bfvx.close()
    with open('bfvy_inception_resnet_v2.pkl', 'wb') as f:
        pickle.dump(y_val1, f)
    # datay = {'features': bottleneck_features_validation, 'labels': y_val1}
    filepathy = 'resnet_validate_bottleneck.json'
    # save_as_pickled_object(data, filepathy)
    # json.dump(data, codecs.open('resnet_validate_bottleneck.json', 'w',encoding='utf-8'),separators=(',', ':'))
    print("json val dump complete")
    # h5x = h5py.File('resnet_validate_bottleneck.h5', 'w')
    # h5x.create_dataset('data', data=data)
print("Reach 4 \n")
def load_bottleneck_data(training_file, validation_file):

    # train_data = try_to_load_as_pickled_object_or_None(training_file)
    # validation_data = try_to_load_as_pickled_object_or_None(training_file)
    # obj_text = codecs.open(training_file, 'r', encoding='utf-8').read()
    # train_data = json.loads(obj_text)

    # obj_text = codecs.open(validation_file, 'r', encoding='utf-8').read()
    # validation_data = json.loads(obj_text)

    # train_data = json.load(data, open(training_file, 'r'))
    # validation_data = json.load(data, open(validation_file, 'r'))
    # X_train = np.array(train_data['features'])
    # y_train = train_data['labels']
    # X_val = np.array(validation_data['features'])
    # y_val = validation_data['labels']
    h5f = h5py.File('bftx_inception_resnet_v2.h5','r')
    X_train2 = h5f['bftx'][:]
    h5f.close()
    h5f = h5py.File('bfvx_inception_resnet_v2.h5','r')
    X_val2 = h5f['bfvx'][:]
    h5f.close()
    with open('bfty_inception_resnet_v2.pkl', 'rb') as f:
        y_train2 = pickle.load(f)
    with open('bfvy_inception_resnet_v2.pkl', 'rb') as f:
        y_val2 = pickle.load(f)
    # X_train2 = datax['features']
    # y_train2 = datax['labels']
    # X_val2 = datay['features']
    # y_val2 = datay['labels']

    return X_train2, y_train2, X_val2, y_val2

X_train, y_train, X_val, y_val = load_bottleneck_data('resnet_train_bottleneck.json',
                                                      'resnet_validate_bottleneck.json')

y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val,num_classes = 10)

# Adding final layer
input_shape = X_train.shape[1:]
inp = Input(shape=input_shape)
x = Flatten()(inp)
x = Dense(num_classes, activation='softmax')(x)
model = Model(inp, x)
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

metrics = Metrics()

with tf.Session() as sess:
    # fetch session so Keras API can work
    K.set_session(sess)
    K.set_learning_phase(1)
    history =model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                       validation_data=(X_val, y_val), shuffle=True, verbose=1,callbacks=[metrics] )
    print(metrics.f1s)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print(history.history['acc'])
    # f1_mean = history.history['f1']
    # val_f1 = history.history['val_f1']

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train ' + str(acc[-1]), 'test ' + str(val_acc[-1])], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train ' + str(loss[-1]), 'test ' + str(val_loss[-1])], loc='upper left')
    plt.show()

    # # summarize history for f1
    plt.plot(metrics.f1s)
    # plt.plot(history.history['val_f1'])
    plt.title('f1 mean score')
    plt.ylabel('f1')
    plt.xlabel('epoch')
    plt.legend(['val ' + str(metrics.f1s[-1])], loc='upper left')
    plt.show()

    print(metrics.confusion[-1])
    ans = input("Do you want to save the weights?")
    if ans == 'y':
        model.save_weights('inception_resnet_v2_bottleneck_weights.h5')
        print("Saved")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
print("Reach End \n")
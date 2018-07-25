from __future__ import print_function
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Input, AveragePooling2D
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
import h5py
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import keras.backend as K

import os
ls1=os.listdir('color')
if '.DS_Store' in ls1:
    ls1.remove('.DS_Store')
dic1={}
for idx,i in enumerate(ls1):
     dic1[i]=idx
import scipy.misc as sm
import numpy as np
count=0
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
h5f = h5py.File('variables.h5','r')
X = h5f['X'][:]
print("Reach 1")
Y = h5f['Y'][:]
print("Reach 2 \n")
batch_size = 16
num_classes = len(dic1)
epochs = 12

# input image dimensions
img_rows, img_cols = 256, 256

h = 224
w = 224
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
        while True:
            # run takes in a tensor/function and performs it.
            # almost always, that function will take a Tensor as input
            # when run is called, it takes a feed_dict param which translates
            # Tensors into actual data/integers/floats/etc
            # this is so you can write a network and only have to change the
            # data being passed in one place instead of everywhere

            # X_batch is resized
            X_batch = session.run(resize_op, {img_placeholder: data[start:end]})
            # X_batch is normalized
            X_batch = preprocess_input(X_batch)
            y_batch = labels[start:end]
            start += batch_size
            end += batch_size
            if start >= n:
                start = 0
                end = batch_size
                print("Bottleneck predictions completed.")

            yield (X_batch, y_batch)

    return _f


def create_model_resnet():
    input_tensor = Input(shape=(h, w, ch))
    model = ResNet50(input_tensor=input_tensor, include_top=False)
    return model
print("Reach 2.8 \n")
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, random_state=0,shuffle=True)
print("Reach 3 \n")
with tf.Session() as sess:
    K.set_session(sess)
    K.set_learning_phase(1)

    model = create_model_resnet()

    train_gen = gen(sess, X_train, y_train, batch_size)
    bottleneck_features_train = model.predict_generator(train_gen(), 2000)
    data = {'features': bottleneck_features_train, 'labels': y_train[:2000]}
    pickle.dump(data, open('resnet_train_bottleneck.p', 'wb'))
    # h5y = h5py.File('resnet_train_bottleneck.h5', 'w')
    # h5y.create_dataset('data',data = data)

    val_gen = gen(sess, X_val, y_val, batch_size)
    bottleneck_features_validation = model.predict_generator(val_gen(), 2000)
    data = {'features': bottleneck_features_validation, 'labels': y_val[:2000]}
    pickle.dump(data, open('resnet_validate_bottleneck.p', 'wb'))
    # h5x = h5py.File('resnet_validate_bottleneck.h5', 'w')
    # h5x.create_dataset('data', data=data)
print("Reach 4 \n")
def load_bottleneck_data(training_file, validation_file):
    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val

X_train, y_train, X_val, y_val = load_bottleneck_data('resnet_train_bottleneck.p',
                                                      'resnet_train_bottleneck.p')

input_shape = X_train.shape[1:]
inp = Input(shape=input_shape)
x = Flatten()(inp)
x = Dense(num_classes, activation='softmax')(x)
model = Model(inp, x)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
with tf.Session() as sess:
    # fetch session so Keras API can work
    K.set_session(sess)
    K.set_learning_phase(1)
    history =model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch_size,
                       validation_data=(X_val, y_val), shuffle=True, verbose=1)
    model.save_weights('resnet_bottleneck_weights.h5')
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

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
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
print("Reach End \n")
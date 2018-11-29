from __future__ import print_function
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Input, AveragePooling2D
from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
import h5py
import matplotlib.pyplot as plt
import pickle
import keras
import numpy as np
from keras import optimizers
import sklearn.metrics as sklm
import tensorflow as tf
import keras.backend as K
import os
from keras.utils.np_utils import to_categorical

# Get number of classes
ls1=os.listdir('color')
if '.DS_Store' in ls1:
    ls1.remove('.DS_Store')
dic1={}
for idx,i in enumerate(ls1):
     dic1[i]=idx

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

# Loading saved predicted X and y
def load_bottleneck_data(training_file, validation_file):

    h5f = h5py.File('bftx_resnet.h5', 'r')
    X_train2 = h5f['bftx'][:]
    h5f.close()
    h5f = h5py.File('bfvx_resnet.h5', 'r')
    X_val2 = h5f['bfvx'][:]
    h5f.close()
    with open('bfty_resnet.pkl', 'rb') as f:
        y_train2 = pickle.load(f)
    with open('bfvy_resnet.pkl', 'rb') as f:
        y_val2 = pickle.load(f)

    return X_train2, y_train2, X_val2, y_val2

# Calling the above function to load saved data
X_train, y_train, X_val, y_val = load_bottleneck_data('resnet_train_bottleneck.json',
                                                      'resnet_validate_bottleneck.json')

y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val,num_classes = 10)

# input image dimensions
img_rows, img_cols = 256, 256
h = 224
w = 224
ch = 3

#HYPERPARAMETERS
batch_size = 128
num_classes = len(dic1)
epochs = 20

#Model
def create_model_resnet():

    input_tensor = Input(shape=(h, w, ch))
    model = ResNet50(input_tensor=input_tensor, include_top=False)
    # x = model.output
    # x = Dropout()(x)
    # model = Model(model.input,x)
    return model

# #Adding final layer

res_model = keras.applications.ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3))
layer_list = [(layer.name) for layer in res_model.layers]
print(layer_list)
top_model = keras.Sequential()
top_model.add(Flatten(input_shape=res_model.output_shape[1:]))
top_model.add(Dense(num_classes,activation='softmax',name='output_pred'))
top_model.load_weights('resnet_bottleneck_weights_001.h5')
layer_list = [(layer.name) for layer in top_model.layers]
print(layer_list)
# model = create_model_resnet()
# input_tensor = Input(shape=(h, w, ch))

model = Model(inputs = res_model.input, outputs = top_model(res_model.output))
layer_list = [(layer.name) for layer in model.layers]
print(layer_list)
model.save_weights('full_model/full_resnet_weights_001.h5')
# input_shape = X_train.shape[1:]
# inp = Input(shape=input_shape)
# x = Flatten()(inp)
# x = Dense(num_classes, activation='softmax')(x)
# model = Model(inp, x)
# model.load_weights('resnet_bottleneck_weights_001.h5')

model_json = model.to_json()
with open("full_model/full_resnet_model.json", "w") as json_file:
    json_file.write(model_json)
# for r,layer in enumerate(layer_list[2:]):
#     # os.makedirs('Intermediate layers/'+ str(r+1) + "_" + str(layer))
#     intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer).output)
#     intermediate_output = intermediate_layer_model.predict(X)
#     # img = deprocess_image(intermediate_output)
#     print(layer)
#     print(intermediate_output[0].shape)
#     # img = np.zeros(intermediate_output[0].shape)
#     # img = intermediate_output[0]
#     # for i in range(img.shape[2]):
#     #     imsave('Intermediate layers/' + str(r+1) + "_" + str(layer) +'/%s_%s.png' % (layer,i+1), img[:,:,i])
# #
# # img_placeholder = tf.placeholder("uint8", intermediate_output.shape)
#
# # tensor. resized images. gets passed into Session()
# # resize_op = tf.image.resize_images(img_placeholder, (224, 224), method=0)
# # img = sess.run(resize_op, {img_placeholder: intermediate_output})
# # print(img.shape)
# # imsave('conv1.png', img)
from __future__ import print_function
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
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
import cv2
from scipy.misc import imsave


def create_model_resnet():

    input_tensor = Input(shape=(224, 224, 3))
    model = ResNet50(input_tensor=input_tensor, include_top=False)
    return model


output_list = ['Tomato___Target_Spot', 'Tomato___Late_blight', 'Tomato___Tomato_mosaic_virus', 'Tomato___Leaf_Mold', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Septoria_leaf_spot']

with tf.Session() as sess:
    K.set_session(sess)
    K.set_learning_phase(1)
    img = image.load_img('1.JPG', target_size=(224, 224))
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    # test = cv2.imread('tomato_mosaic.jpg')
    # test = cv2.cvtColor(test,cv2.COLOR_BGR2RGB)
    # # test = image.img_to_array(test)
    # test = cv2.flip(test,0)
    # imsave('/Volumes/My Passport/img_test.png', test)
    X = preprocess_input(X)
    # model = create_model_resnet()
    # layer_dict = dict([(layer.name,layer) for layer in model.layers])
    # layer_list = [(layer.name) for layer in model.layers]
    # print(layer_list)
    # y = model.predict(X)
    # print(y.shape)
    # json_file = open('resnet_model.json','r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # final_model = keras.models.model_from_json(loaded_model_json)
    # final_model.load_weights('resnet_bottleneck_weights_001.h5')
    # Y = final_model.predict(y)
    json_file = open('full_model/full_resnet_model.json','r')
    model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(model_json)
    model.load_weights('full_model/full_resnet_weights_001.h5')

    Y = model.predict(X)
    print(Y)
    print(output_list[np.argmax(Y)])

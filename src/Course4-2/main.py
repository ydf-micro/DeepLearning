# *_*coding:utf-8 *_*

import numpy as np
from keras import layers
from keras.layers import *
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import kt_utils

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def model(input_shape):
    '''

    :param input_shape:
    :return:
    '''

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool')(X)

    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model

def HappyModel(input_shape):
    '''

    :param input_shape:
    :return:
    '''

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool')(X)

    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model

if __name__ == '__main__':
    # X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = kt_utils.load_dataset()
    #
    # X_train = X_train_orig / 255
    # X_test = X_test_orig / 255
    #
    # Y_train = Y_train_orig.T
    # Y_test = Y_test_orig.T
    #
    # print(X_train.shape[0])
    # print(X_test.shape[0])
    #
    # print(X_train.shape)
    # print(Y_train.shape)
    # print(X_test.shape)
    # print(Y_test.shape)
    #
    # happy_model = HappyModel(X_train.shape[1:])
    # happy_model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    #
    # happy_model.fit(X_train, Y_train, epochs=40, batch_size=50)
    # happy_model.save('models/happymodel.h5')
    #
    # preds = happy_model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)
    #
    # print(preds[0], preds[1])

    img_path = 'images/angry.jpeg'
    img = image.load_img(img_path, target_size=(64, 64))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    happy_model = load_model('models/happymodel.h5')
    print(happy_model.predict(x))
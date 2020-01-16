import numpy as np
from matplotlib import pyplot as plt

import keras
import keras.backend as K
from keras import models
from keras import layers
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import tensorflow as tf

from ista import ISTA

###
#$$ nonconvex loss function
###
def not_convex(y_true, y_pred):
    return K.sum(K.square(y_true - y_pred)) / ( K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) )

###
### start a session - will need same session to link K to tf
###

session = tf.Session()
K.set_session(session)

###
### generate dummy model
###

(x_train, y_train), (x_test, y_test) = mnist.load_data()

train = x_train.reshape((60000,28*28)).astype('float32') / 255
test  =  x_test.reshape((10000,28*28)).astype('float32') / 255

train_labels = to_categorical(y_train)
test_labels  = to_categorical(y_test)

mu = 0.0001
constraints = [keras.regularizers.l1(mu), ISTA(mu)]
locations   = ['l1', 'ista']

for con, loc in zip(constraints, locations):

    net = models.Sequential()
    net.add(layers.Dense(256, activation='relu', input_shape=(28*28,)))
    net.add(layers.Dense(128, activation='relu'))
    if loc == 'l1':
        net.add(layers.Dense(128, activation='relu', kernel_regularizer=con))
    elif loc == 'ista':
        net.add(layers.Dense(128, activation='relu', kernel_constraint=con))
    else:
        net.add(layers.Dense(128, activation='relu'))
    net.add(layers.Dense(10, activation='softmax'))
    net.compile(optimizer='adam',loss=not_convex,metrics=['accuracy'])
    # net.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    h = net.fit(train,
        train_labels,
        epochs=40,
        batch_size=128,
        shuffle=True)

    net.save('mnistmodel-'+loc+'.h5')

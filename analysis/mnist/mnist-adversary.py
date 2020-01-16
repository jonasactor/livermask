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

from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod

from ista import ISTA

###
### start a session - will need same session to link K to tf
###

session = tf.Session()
K.set_session(session)

###
### generate dummy model
###

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#mu = 0.00
mu = 0.001  # proxy for some form of spectral regularization, even though there's no conv here

net = models.Sequential()
net.add(layers.Dense(256, activation='relu', input_shape=(28*28,)))
net.add(layers.Dense(128, activation='relu'))
net.add(layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l1(mu)))
# net.add(layers.Dense(128, activation='relu', kernel_constraint=ISTA(mu)))
net.add(layers.Dense(10, activation='softmax'))
net.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_crossentropy', 'accuracy'])

train = x_train.reshape((60000,28*28)).astype('float32') / 255
test  =  x_test.reshape((10000,28*28)).astype('float32') / 255

train_labels = to_categorical(y_train)
test_labels  = to_categorical(y_test)

h = net.fit(train,
    train_labels,
    epochs=40,
    batch_size=128,
    shuffle=True,
    callbacks=[ModelCheckpoint('mnistmodel.h5',save_best_only=True)],
    validation_data=(test, test_labels))

net.save('mnistmodel.h5')

###
### load model and perform adversarial attack
###

net = load_model('mnistmodel.h5', custom_objects={'ISTA':ISTA})
wrapper = KerasModelWrapper(net)

x = tf.placeholder(tf.float32, shape=(None, 784)) # mnist
y = tf.placeholder(tf.float32, shape=(None,  10)) # mnist

fgsm = FastGradientMethod(wrapper, sess=session)

fgsm_eps = 0.1
fgsm_min = 0.0
fgsm_max = 1.0
fgsm_parameters = {'eps':fgsm_eps, 'clip_min':fgsm_min, 'clip_max':fgsm_max }
adversary = fgsm.generate(x, **fgsm_parameters)
adversary = tf.stop_gradient(adversary)
adv_prob = net(adversary)

fetches = [adv_prob]
fetches.append(adversary)
outputs = session.run(fetches=fetches, feed_dict={x:test})
adv_prob = outputs[0]
adv_examples = outputs[1]
adv_predicted = adv_prob.argmax(1)
adv_accuracy = np.mean(adv_predicted==y_test)
print("accuracy:\t %.5f" % adv_accuracy)

n_classes = 10
f, ax = plt.subplots(2,5,figsize=(10,5))
ax = ax.flatten()
for i in range(n_classes):
    diff = adv_examples[i] - test[i]
    norm_diff = np.linalg.norm(diff)
    max_diff  = np.abs(diff).max()
    print('norm: ', norm_diff, ' max abs val: ', max_diff)
    ax[i].imshow(diff.reshape(28,28))
plt.show()

f,ax = plt.subplots(2,5,figsize=(10,5))
ax = ax.flatten()
for i in range(n_classes):
    ax[i].imshow(adv_examples[i].reshape(28,28))
    ax[i].set_title("adv: %d, label: %d" % (adv_predicted[i], y_test[i]))
plt.show()

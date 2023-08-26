import tensorflow as tf
import numpy as np
from tensorflow import keras


class Network(tf.keras.Model):

    def __init__(self, action_dim):
        super(Network,self).__init__()
        #Convolutions on the frames on the screen
        conv1 = keras.layers.Conv2D(32,8,strides=4,activation="relu")
        conv2 = keras.layers.Conv2D(64,4,strides=2,activation="relu")
        conv3 = keras.layers.Conv2D(64,3,strides=1,activation="relu")

        flatten = keras.layers.Flatten()

        dense1 = keras.layers.Dense(512, activation="relu")
        output = keras.layers.Dense(self.action_dim, activation="linear")

    def call(self):
        inputs = keras.layers.Input(shape=(84,84,4,))

        #Convolutions on the frames on the screen
        conv1 = keras.layers.Conv2D(32,8,strides=4,activation="relu")(inputs)
        conv2 = keras.layers.Conv2D(64,4,strides=2,activation="relu")(conv1)
        conv3 = keras.layers.Conv2D(64,3,strides=1,activation="relu")(conv2)

        flatten = keras.layers.Flatten()(conv3)

        dense1 = keras.layers.Dense(512, activation="relu")(flatten)
        output = keras.layers.Dense(self.action_dim, activation="linear")(dense1)

        return keras.Model(inputs = inputs, outputs=output)
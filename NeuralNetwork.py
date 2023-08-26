import tensorflow as tf
import numpy as np
from tensorflow import keras


class Network(tf.keras.Model):

    def __init__(self, action_dim):
        super(Network,self).__init__()
        self.conv1 = keras.layers.Conv2D(32,(8,8), strides=(4,4), activation='relu')
        self.conv2 = keras.layers.Conv2D(64,(4,4), strides=(2,2), activation='relu')
        self.conv3 = keras.layers.Conv2D(64,(3,3), activation='relu')
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(512,activation='relu')
        self.output_layer = keras.layers.Dense(action_dim, activation='linear')

    def call(self, inputs):
        inputs = keras.Input(shape=(84,84,4,))
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.output_layer(x)
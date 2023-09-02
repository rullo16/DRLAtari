from keras.src.utils import losses_utils
import tensorflow as tf
from tensorflow import keras

class MSE(keras.losses.Loss):

    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        return tf.mean(tf.square(y_true - y_pred))


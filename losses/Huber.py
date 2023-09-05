import tensorflow as tf
from tensorflow import keras


class Huber(keras.losses.Loss):

    def __init__(self, threshold = 0.8):
        super(Huber,self).__init__()
        self.threshold = threshold

    def call(self,y_true, y_pred):

        error = y_true - y_pred
        low_err = tf.abs(error)<=self.threshold
        low_err_loss = tf.square(error) / 2
        high_err_loss = self.threshold * (tf.abs(error)-self.threshold/2)
        return tf.where(low_err,low_err_loss,high_err_loss)


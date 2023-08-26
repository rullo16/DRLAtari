import tensorflow as tf
from tensorflow import keras


class Huber(keras.losses.Loss):

    def __init__(self, threshold = 0.8):
        super(Huber,self).__init__()
        self.threshold = threshold

    def call(self,y_true, y_pred):

        error = y_true - y_pred
        small_err = tf.abs(error)<=self.threshold
        small_err_loss = tf.square(error) / 2
        big_err_loss = self.threshold * (tf.abs(error)-self.threshold/2)
        return tf.where(small_err,small_err_loss,big_err_loss)


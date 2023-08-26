from keras.src.utils import losses_utils
import tensorflow as tf
from tensorflow import keras

class MSE(keras.losses.Loss):

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name=None):
        super().__init__(reduction, name)

    def call(self, y_true, y_pred):
        
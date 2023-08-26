from keras.src.utils import losses_utils
import tensorflow as tf
from tensorflow import keras

class RMSProp(keras.optimizers.Optimizer):

    def __init__(self, name, weight_decay=0, clipnorm=None, clipvalue=None, global_clipnorm=None, use_ema=False, ema_momentum=0.99, ema_overwrite_frequency=None, jit_compile=True, **kwargs):
        super().__init__(name, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, jit_compile, **kwargs)
        
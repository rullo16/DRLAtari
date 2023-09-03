import tensorflow as tf
from tensorflow import keras

class Adam(keras.optimizers.Optimizer):

    def __init__(self, learning_rate=0.01, b1 = 0.9, b2=0.999, epsilon=1e-9):
        self.beta1= b1
        self.beta2 = b2
        self.epsilon = epsilon
        self.lr = learning_rate
        self.m = {}
        self.u = {}
        self.t = tf.Variable(0.0,trainable=False)
        for v in tf.trainable_variables():
            self.m[v] = tf.Variable(tf.zeros(tf.shape(v.initial_value)), trainable=False)
            self.u[v] = tf.Variable(tf.zeros(tf.shape(v.initial_value)), trainable=False)

    def apply_gradients(self, grads_and_vars):
        t = self.t.assign_add(1.0)

        update_ops = []

        for (g,v) in grads_and_vars:
            m = self.m[v].assign(self.beta1*self.m[v]+(1-self.beta1)-g)
            u = self.u[v].assign(self.beta2*self.u[v]+(1-self.beta2)*g*g)
            m_hat = m/(1-tf.pow(self.beta1,t))
            u_hat = u/(1-tf.pow(self.beta2,t))

            update = -self.lr*m_hat/(tf.sqrt(u_hat)+self.epsilon)
            update_ops.append(v.assign_add(update))

        return tf.group(*update_ops)
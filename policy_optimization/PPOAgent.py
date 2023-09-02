import numpy as np
from tensorflow import keras
import tensorflow as tf
from losses.Huber import Huber
import random
from collections import deque
from NeuralNetwork import QNetworkModel


class PPOAgent:

    def __init__(self, action_dim, epsilon, value_coeff, entropy_coeff, logging):

        self.action_dim = action_dim
        self.epsion = epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        
        self.actor = QNetworkModel(self.action_dim)
        self.critic = QNetworkModel(1)

        self.batch_size = 32

        

        self.optimizer_actor = keras.optimizers.AdamW(learning_rate=0.0001)
        self.optimizer_critic = keras.optimizers.AdamW(learning_rate=0.0002)
        self.loss_function = Huber()

        self.buffer = deque(maxlen=10000)

        self.training_errors = []
        self.logging = logging

    def get_action(self,obs):
        obs_tensor = tf.convert_to_tensor(obs)
        obs_tensor = tf.expand_dims(obs_tensor,0)
        obs_tensor = tf.transpose(obs_tensor,(0,2,3,1))
        logits = self.actor.predict(obs_tensor)[0]
        action_probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits-np.max(logits)))
        action = np.random.choice(self.action_dim, p=action_probs)
        return action, action_probs[action]
    
    def get_value(self,obs):
        obs_tensor = tf.convert_to_tensor(obs)
        obs_tensor = tf.expand_dims(obs_tensor,0)
        obs_tensor = tf.transpose(obs_tensor,(0,2,3,1))
        return self.critic.predict(obs_tensor)[0][0]
    
    def update(self,obs,actions,advantages,old_action_probs):
        obs = np.transpose(obs,(0,2,3,1))
        with tf.GradientTape(persistent=True) as tape:
            new_action_probs = self.actor.call(obs)
            value_preds = self.critic.call(obs)

            ratio = new_action_probs / old_action_probs
            surrogate1 = ratio * advantages
            surrogate2 = tf.clip_by_value(ratio, 1-self.epsion, 1+self.epsion) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1,surrogate2))
            value_loss = tf.reduce_mean(tf.square(value_preds - advantages))
            entropy = -tf.reduce_mean(new_action_probs*tf.math.log(new_action_probs))
            total_loss = (actor_loss+self.value_coeff*value_loss - self.entropy_coeff*entropy)
        
        grads_actor = tape.gradient(total_loss, self.actor.trainable_variables)
        grads_critic = tape.gradient(value_loss, self.critic.trainable_variables)

        self.optimizer_actor.apply_gradients(zip(grads_actor,self.actor.trainable_variables))
        self.optimizer_critic.apply_gradients(zip(grads_critic, self.critic.trainable_variables))

        return total_loss.numpy()
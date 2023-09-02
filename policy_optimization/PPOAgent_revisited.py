import numpy as np
from tensorflow import keras
import tensorflow as tf
from losses.Huber import Huber
import random
from collections import deque
from NeuralNetwork import QNetworkModel

class Actor:

    def __init__(self,action_dim) -> None:
        self.action_dim = action_dim
        self.model = QNetworkModel(self.action_dim)
        self.optimizer = tf.keras.optimizers.AdamW(learning_rate=0.005)
        self.epsilon = 0.1

    def compute_loss(self, old_policy, new_policy, actions, gaes):
        gaes = tf.stop_gradient(gaes)
        old_log_prob = tf.stop_gradient(tf.math.log(tf.reduce_sum(old_policy*actions)))
        log_prob = tf.math.log(tf.reduce_sum(new_policy*actions))
        ratio = tf.math.exp(log_prob-old_log_prob)
        clipped_ratio = tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon)
        surrogate = -tf.minimum(ratio*gaes, self.epsilon*gaes)
        return tf.reduce_mean(surrogate)
    
    def train(self, old_policy, obs, actions, gaes):
        actions = tf.one_hot(actions, self.action_dim)
        actions = tf.reshape(actions, [-1, self.action_dim])
        actions = tf.cast(actions, tf.float64)

        with tf.GradientTape() as tape:
            logits = self.model(obs, training=True)
            loss = self.compute_loss(old_policy, logits, actions, gaes)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
    

class Critic:

    def __init__(self) -> None:
        self.action_dim = 1
        self.model = QNetworkModel(self.action_dim)
        self.optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)
        self.epsilon = 0.1

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MSE()
        return mse(td_targets, v_pred)
    
    def train(self, obs, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(obs, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred,tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Agent:
    def __init__(self, action_dim) -> None:
        self.action_dim = action_dim
        
        self.actor = Actor(self.action_dim)
        self.critic = Critic()

        self.gamma = 0.99
        self.lam = 0.95

    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.gamma * forward_val - v_values[k]
            gae_cumulative = self.gamma * self.lam * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        
        return gae, n_step_targets
    
    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch
    
    def get_action(self, obs):
        obs_tensor = tf.convert_to_tensor(obs)
        obs_tensor = tf.expand_dims(obs_tensor,0)
        obs_tensor = np.transpose(obs_tensor,(0,2,3,1))
        probs = self.actor.model.predict(obs_tensor)
        return np.random.choice(self.action_dim, p = probs[0]), probs
    

        
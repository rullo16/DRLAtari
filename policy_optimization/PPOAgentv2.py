import sys
sys.path.append('../')

import numpy as np
from tensorflow import keras
import tensorflow as tf
from losses.Huber import Huber
import random
from collections import deque
from NeuralNetwork import QNetworkModel
import tensorflow_probability as tfp

class PPOAgent:

    def __init__(self, action_dim):
        self.actor = QNetworkModel(action_dim)
        self.critic = QNetworkModel(1)

        self.actor_optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-5)
        self.critic_optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-5)

        self.clip = 0.2

    def act(self, obs):
        obs_tensor = tf.convert_to_tensor(obs)
        action_prob = self.actor.call(obs_tensor).numpy()
        distribution = tfp.distributions.Categorical(probs=action_prob, dtype=tf.float32)
        action = distribution.sample()
        
        return int(action.numpy()[0])
    
    def GAE(self, obs, actions, rewards, done, values, gamma):
        g = 0
        lmb = 0.95
        returns = []

        for i in reversed(range(len(rewards))):
            delta = rewards[i]+gamma*values[i+1]*done[i]-values[i]
            g = delta + gamma * lmb * done[i] * g
            returns.append(g+values[i])

        returns.reverse()
        advantage = np.array(returns, dtype=np.float32)-values[:-1]
        advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + 1e-10)
        obs = np.array(obs, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)

        return obs, actions, returns, advantage
    
    def actor_loss(self, probs, actions, advantage, old_probs, critic_loss):

        probability = probs
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability, tf.math.log(probability))))

        sur1 = []
        sur2 = []

        for pb, t, op, a in zip(probability, advantage, old_probs, actions):
            t = tf.constant(t)
            ratio = tf.math.divide(pb[a],op[a])
            s1 = tf.math.multiply(ratio,t)
            s2 = tf.math.multiply(tf.clip_by_value(ratio, 1.0 -self.clip, 1.0 + self.clip),t)

            sur1.append(s1)
            sur2.append(s2)
        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)

        loss=tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1,sr2))-critic_loss+0.001*entropy)
        return loss

    
    def learn(self, obs, actions, advantage, old_probs, distinct_r):
        distinct_r = tf.reshape(distinct_r, (len(distinct_r),))
        advantage = tf.reshape(advantage,(len(advantage),))

        old_p = old_probs

        # old_p = tf.reshape(old_p, (len(old_p),2))

        obs = np.squeeze(obs,axis=1)
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            prob = self.actor.call(obs,training=True)
            v = self.critic.call(obs, training=True)
            v = tf.reshape(v, (len(v),))
            temporal_difference = tf.math.subtract(distinct_r,v)
            critic_loss = 0.5 * keras.losses.MSE(distinct_r,v)
            actor_loss = self.actor_loss(prob, actions, advantage, old_p, critic_loss)

        grads1 = tape1.gradient(actor_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(critic_loss, self.actor.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(grads1,self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(grads2,self.critic.trainable_variables))

        return actor_loss, critic_loss
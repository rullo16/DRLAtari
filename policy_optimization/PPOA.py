import sys
sys.path.append('../')

import numpy as np
from tensorflow import keras
import tensorflow as tf
from losses.Huber import Huber
import random
from collections import deque
from NeuralNetwork import PPONetworkModel
import tensorflow_probability as tfp


class PPOAgent:

    def __init__(self,action_dim):
        
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon=0.1
        self.learning_rate = 2.5e-4
        self.lam = 0.95
        self.vf_coeff = 1
        self.entropy_coeff = 0.01

        self.actor = PPONetworkModel(action_dim)
        self.critic = PPONetworkModel(1)


        self.actor_optimizer = keras.optimizers.AdamW(learning_rate=self.learning_rate)
        self.critic_optimizer = keras.optimizers.AdamW(learning_rate=self.learning_rate)

    def convert_to_tensor_observations(self,obs):
        obs_tensor = tf.convert_to_tensor(obs)
        obs_tensor = tf.expand_dims(obs_tensor,0)
        obs_tensor = tf.transpose(obs_tensor,(0,2,3,1))
        return obs_tensor
    
    def act(self,obs):
        obs = self.convert_to_tensor_observations(obs)
        action_prob = self.actor.predict(obs)[0]
        action = np.random.choice(self.action_dim, p=action_prob)
        return action, action_prob[action]
    
    def get_advantage(self, reward,done,next_obs,obs):
        obs = self.convert_to_tensor_observations(obs)
        next_obs = self.convert_to_tensor_observations(next_obs)

        discounted_reward = reward + self.gamma * (1-done) * self.critic.predict(next_obs)
        baseline = self.critic.predict(obs)
        return discounted_reward - baseline
    
    def train(self, obs, actions, old_probs, advantages, returns):
        #obs = self.convert_to_tensor_observations(obs)
        if not tf.is_tensor(obs):
            obs = tf.convert_to_tensor(obs)
            obs = tf.squeeze(obs,axis=1)
        if not tf.is_tensor(actions):
            actions = tf.convert_to_tensor(actions)
        if not tf.is_tensor(old_probs):
            old_probs = tf.convert_to_tensor(old_probs)
        if not tf.is_tensor(advantages):
            advantages = tf.convert_to_tensor(advantages)
        if not tf.is_tensor(returns):
            returns = tf.convert_to_tensor(returns)
        
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            #Calculate action probabilities using the actor network
            action_probs = self.actor.call(obs)
            #Calculate the probabilities of the chosen action
            new_probs = tf.reduce_sum(actions * action_probs, axis=1)
            #Calculate the ratio of new and old action probabilities
            ratio = new_probs / old_probs
            #Calculate the surrogate loss for the policy update
            surrogate = ratio * advantages

            #Clipping to ensure stability in policy updates
            surrogate2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            # Calculate actor loss as the negative of the minimum of the surrogates, negative to convert it into a minimization problem
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate, surrogate2))

            #Calculate the entropy of action probabilities for entropy regularization
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1)
            #Calculate the entropy loss as the negative of the mean of the entropy, negative because the goal is to minimize entropy.
            #When entropy is high the policy explores more.
            entropy_loss = -self.entropy_coeff * tf.reduce_mean(entropy)

            #Calculated the values predicted by the critic network
            values = self.critic.call(obs)
            #Calculate the value function loss
            critic_loss = tf.reduce_mean(tf.square(returns - values))
            
            vf_loss = self.vf_coeff * critic_loss
            


            total_loss = actor_loss + critic_loss + vf_loss + entropy_loss

        actor_gradients = tape1.gradient(total_loss, self.actor.trainable_variables())
        critic_gradients = tape2.gradient(total_loss, self.critic.trainable_variables())

        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables()))
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables()))

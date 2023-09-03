import sys
sys.path.append('../')

import numpy as np
from tensorflow import keras
import tensorflow as tf
from losses.Huber import Huber
import random
from collections import deque
from NeuralNetwork import QNetworkModel

class DoubleDQNAgent:

    def __init__(self, action_dim, learning_rate, gamma, epsilon_decay):
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = epsilon_decay

        self.batch_size = 32

        #First model makes the predictions for Q-values which are then used to make an action
        self.net = QNetworkModel(self.action_dim)
        #We use a target model for the prediction of future rewards. The weights gets updated over n number of steps
        #So that when we calculate the loss between the Q-values, the target Q-value is stable.
        self.net_target = QNetworkModel(self.action_dim)

        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_function = Huber()

        self.buffer = deque(maxlen=10000)

        self.training_errors = []

    def update_target_net(self):
        self.net_target.update_weights(self.net.get_weights())

    def remember(self, obs,action,reward,next_state,done):
        #Save Actions and states in memory to be used in the replay buffer
        self.buffer.append([obs,action,reward,next_state, done])

    def action_selection(self,obs,frame_count):
        #Explore
        if np.random.random() < self.epsilon or self.epsilon<frame_count:
            return np.random.choice(self.action_dim)
        obs_tensor = tf.convert_to_tensor(obs)
        obs_tensor = tf.expand_dims(obs_tensor,0)
        q_value = tf.argmax(self.net.predict(obs_tensor)[0]).numpy()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return q_value
        

    def replay(self):
        sample = random.sample(self.buffer, self.batch_size)
        obs, actions, rewards, next_obs, done = map(np.asarray, zip(*sample))
        obs = np.transpose(obs,(0,2,3,1))
        next_obs = np.transpose(next_obs,(0,2,3,1))

        next_q_vals = tf.reduce_max(self.net_target.predict(next_obs),axis=1)
        updated_q_vals = rewards + next_q_vals * self.gamma
        targets = updated_q_vals * (1-done)

        masks = tf.one_hot(actions,self.action_dim)

        with tf.GradientTape() as tape:

            q_vals = self.net.call(obs)
            q_action = tf.reduce_sum(tf.multiply(q_vals,masks),axis=1)
            loss = self.loss_function(targets,q_action)

        self.training_errors.append(loss.numpy())
        #Backpropagation
        grads = tape.gradient(loss, self.net.trainable_variables())
        self.optimizer.apply_gradients(zip(grads,self.net.trainable_variables()))
    
    

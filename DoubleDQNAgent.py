import numpy as np
import tensorflow as tf
from tensorflow import keras
from NeuralNetwork import Network
from losses import Huber
from collections import deque
import wandb


class DoubleDQNAgent:

    def __init__(self, action_dim, learning_rate, gamma, epsilon_decay) -> None:
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = epsilon_decay
        
        self.batch_size = 32

        self.network = Network()

        self.target_net = Network()

        self.optimizer = keras.optimizers.AdamW(learning_rate=self.learning_rate)
        self.loss_function = Huber.Huber()

        self.buffer = deque(maxlen=10000)

        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []
        self.episode_reward_history = []
        self.training_error = []

    def set_weights(self):
        weights = self.network.model.get_weights()
        self.target_net.set_weights(weights)
    
    def action_selection(self, obs, frame_count):
        #Explore
        if np.random.random() < self.epsilon or self.epsilon<frame_count:
            return np.random.choice(self.action_dim)
        #Best Action
        state_tensor = tf.convert_to_tensor(obs)
        state_tensor = tf.expand_dims(state_tensor,0)
        action_probs = self.net(state_tensor,training=False)
        action = tf.argmax(action_probs[0]).numpy()

        #Update the probability of taking random actions
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action
    
    def remember(self, obs,action,reward,next_state,done):
        #Save Actions and states in memory to be used in the replay buffer
        self.buffer.append([obs,action,reward,next_state, done])


    def replay(self):
        #Update the network based on the experience
        if len(self.done_history)<self.batch_size:
            return
        
        #Get indices of samples fro replay buffers
        sample = np.random.sample(self.buffer, self.batch_size)
        obs_sample, action_sample, rewards_sample, next_obs_sample, done_sample = map(np.asarray, zip(*sample))
        
        #Using list comprehension to sample from replay buffer
        obs_sample = np.transpose(obs_sample,(0,2,3,1))
        next_obs_sample = np.transpose(next_obs_sample,(0,2,3,1))

        done_sample = tf.convert_to_tensor(done_sample)

        #Build the updated Q-values for the sampled future states, we use the target model for stability
        future_rewards = self.target_net.predict(next_obs_sample)
        #Q-value = reward + discount_factor*expected future reward
        update_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards,axis=1)
        #If final frame set the last value to -1
        update_q_values = update_q_values * (1-done_sample)- done_sample

        #We create a mask to calculate the loss
        masks = tf.one_hot(action_sample,self.action_dim)

        with tf.GradientTape() as tape:
            #Train the model on the states and updated Q-values
            q_values = self.net(obs_sample)
            #Apply the mask to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            #calculate the loss between old and new Q-values
            loss = self.loss_function(update_q_values,q_action)
        self.training_error.append(loss.numpy())
        #Backpropagation
        grads = tape.gradient(loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.net.trainable_variables))
        
        
    def update_target_net(self):
        #Update the target network with new weights
        self.net_target.set_weights(self.net.get_weights())
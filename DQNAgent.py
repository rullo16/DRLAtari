import numpy as np
from tensorflow import keras
import tensorflow as tf

class DQNAgent:
    def __init__(self, action_dim, learning_rate, gamma, epsilon_decay):
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = epsilon_decay

        self.batch_size = 32

        #First model makes the predictions for Q-values which are then used to make an action
        self.net = self.create_q_model()
        #We use a target model for the prediction of future rewards. The weights gets updated over n number of steps
        #So that when we calculate the loss between the Q-values, the target Q-value is stable.
        self.net_target = self.create_q_model()

        self.optimizer = keras.optimizers.AdamW(learning_rate=self.learning_rate)
        self.loss_function = keras.losses.Huber()

        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []
        self.episode_reward_history = []
        self.training_error = []
    
    def create_q_model(self):
        #Network defined by the DeepMind Paper
        inputs = keras.layers.Input(shape=(84,84,4,))

        #Convolutions on the frames on the screen
        conv1 = keras.layers.Conv2D(32,8,strides=4,activation="relu")(inputs)
        conv2 = keras.layers.Conv2D(64,4,strides=2,activation="relu")(conv1)
        conv3 = keras.layers.Conv2D(64,3,strides=1,activation="relu")(conv2)

        flatten = keras.layers.Flatten()(conv3)

        dense1 = keras.layers.Dense(512, activation="relu")(flatten)
        output = keras.layers.Dense(self.action_dim, activation="linear")(dense1)

        return keras.Model(inputs = inputs, outputs=output)

    def action_selection(self, obs,frame_count):
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
        self.action_history.append(action)
        self.state_history.append(obs)
        self.state_next_history.append(next_state)
        self.done_history.append(done)
        self.rewards_history.append(reward)

    def replay(self):
        #Update the network based on the experience
        if len(self.done_history)<self.batch_size:
            return
        
        #Get indices of samples fro replay buffers
        indices = np.random.choice(range(len(self.done_history)), size=self.batch_size)
        
        #Using list comprehension to sample from replay buffer
        state_sample = np.array([self.state_history[i] for i in indices])
        state_sample = np.transpose(state_sample,(0,2,3,1))

        state_next_sample = np.array([self.state_next_history[i] for i in indices])
        state_next_sample = np.transpose(state_next_sample,(0,2,3,1))

        rewards_sample = [self.rewards_history[i] for i in indices]
        action_sample = [self.action_history[i] for i in indices]
        done_sample = tf.convert_to_tensor([float(self.done_history[i]) for i in indices])

        #Build the updated Q-values for the sampled future states, we use the target model for stability
        future_rewards = self.net_target.predict(state_next_sample)
        #Q-value = reward + discount_factor*expected future reward
        update_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards,axis=1)
        #If final frame set the last value to -1
        update_q_values = update_q_values * (1-done_sample)- done_sample

        #We create a mask to calculate the loss
        masks = tf.one_hot(action_sample,self.action_dim)

        with tf.GradientTape() as tape:
            #Train the model on the states and updated Q-values
            q_values = self.net(state_sample)
            #Apply the mask to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            #calculate the loss between old and new Q-values
            loss = self.loss_function(update_q_values,q_action)
        self.training_error.append(loss.numpy())
        #Backpropagation
        grads = tape.gradient(loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.net.trainable_variables))

        #Limit the observation and reward history
        if len(self.rewards_history) > 100000:
            del self.rewards_history[:1]
            del self.state_history[:1]
            del self.action_history[:1]
            del self.done_history[:1]
            del self.state_next_history[:1]
        


    def update_target_net(self):
        #Update the target network with new weights
        self.net_target.set_weights(self.net.get_weights())

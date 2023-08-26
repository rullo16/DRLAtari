import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

class DoubleQNAgent:

    def __init__(self, action_dim, learning_rate, gamma, epsilon_decay) -> None:
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = epsilon_decay
        
        self.batch_size = 32

        self.network = self.create_q_model()

        self.target_net = self.create_q_model()

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

        inputs = keras.layers.Input(shape=(84,84,4,))

        #Convolutions on the frames on the screen
        conv1 = keras.layers.Conv2D(32,8,strides=4,activation="relu")(inputs)
        conv2 = keras.layers.Conv2D(64,4,strides=2,activation="relu")(conv1)
        conv3 = keras.layers.Conv2D(64,3,strides=1,activation="relu")(conv2)

        flatten = keras.layers.Flatten()(conv3)

        dense1 = keras.layers.Dense(512, activation="relu")(flatten)
        output = keras.layers.Dense(self.action_dim, activation="linear")(dense1)

        return keras.Model(inputs = inputs, outputs=output)

    
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
from collections import defaultdict
import numpy as np

class QLearningAgent:
    def __init__(self, env,learning_rate, initial_epsilon, epsilon_decay, final_epsilon, discount_factor):
        
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.learning_rate = learning_rate
        self.gamma = discount_factor
        
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_errors = []
        
    def get_action(self, obs):
        if np.random.random()< self.epsilon:
            return self.env.action_space.sample()
        
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(self,obs,action,reward, terminated, next_obs):
        future_q_value = np.max(self.q_values[next_obs])
        temporal_difference = (reward+self.gamma*future_q_value-self.q_values[obs][action])
        self.q_values[obs][action] = (self.q_values[obs][action]+self.learning_rate*temporal_difference)
        self.training_errors.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon,self.epsilon-self.epsilon_decay)
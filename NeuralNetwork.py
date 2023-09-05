import tensorflow as tf
from tensorflow import keras


class QNetworkModel:

    def __init__(self,action_dim):
        self.action_dim = action_dim
        
        self.model = self.create_model()
    
    def create_model(self):
        # Network defined by the Deepmind paper
        inputs = keras.layers.Input(shape=(84, 84, 4,))

        # Convolutions on the frames on the screen
        x = keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        x = keras.layers.Conv2D(64, 4, strides=2, activation="relu")(x)
        x = keras.layers.Conv2D(64, 3, strides=1, activation="relu")(x)

        x = keras.layers.Flatten()(x)

        x = keras.layers.Dense(512, activation="relu")(x)
        action = keras.layers.Dense(self.action_dim, activation="linear")(x)

        return keras.Model(inputs=inputs, outputs=action)
    
    def predict(self, obs):
        return self.model.predict(obs)
    
    def call(self, obs,training=False):
        return self.model(obs,training=training)
    
    def train(self, obs, target):
        self.model.fit(obs,target, epochs=1, verbose=0)

    def trainable_variables(self):
        return self.model.trainable_variables
    
    def update_weights(self,weights):
        self.model.set_weights(weights=weights)

    def get_weights(self):
        return self.model.get_weights()
    
    def save_model(self,name):
        self.model.save(name)
    
    def load_model(self,name):
        return keras.models.load_model(name)

class DuelingQNetworkModel:

    def __init__(self,action_dim):
        self.action_dim = action_dim
        self.model = self.create_dueling_model()
    
    def create_dueling_model(self):
        # Network defined by the Deepmind paper
        inputs = keras.layers.Input(shape=(84, 84, 4,))

        # Convolutions on the frames on the screen
        x = keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        x = keras.layers.Conv2D(64, 4, strides=2, activation="relu")(x)
        x = keras.layers.Conv2D(64, 3, strides=1, activation="relu")(x)

        x = keras.layers.Flatten()(x)

        x = keras.layers.Dense(512, activation="relu")(x)

        #Advantage and Value Streams
        advantage = keras.layers.Dense(self.action_dim, activation="linear")(x)
        value = keras.layers.Dense(1, activation="linear")(x)

        action = keras.layers.Add()([advantage,value])

        return keras.Model(inputs=inputs, outputs=action)
    
    def predict(self, obs):
        return self.model.predict(obs)
    
    def call(self, obs,training=False):
        return self.model(obs,training=training)
    
    def train(self, obs, target):
        self.model.fit(obs,target, epochs=1, verbose=0)

    def trainable_variables(self):
        return self.model.trainable_variables
    
    def update_weights(self,weights):
        self.model.set_weights(weights=weights)

    def get_weights(self):
        return self.model.get_weights()
    
    def save_model(self,name):
        self.model.save(name)
    
    def load_model(self,name):
        return keras.models.load_model(name)

class PPONetworkModel:

    def __init__(self,action_dim):
        self.action_dim = action_dim
        self.model = self.create_model()
    
    def create_model(self):
        # Network defined by the Deepmind paper
        inputs = keras.layers.Input(shape=(84, 84, 4,))

        # Convolutions on the frames on the screen
        x = keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        x = keras.layers.Conv2D(64, 4, strides=2, activation="relu")(x)
        x = keras.layers.Conv2D(64, 3, strides=1, activation="relu")(x)

        x = keras.layers.Flatten()(x)

        x = keras.layers.Dense(512, activation="relu")(x)
        action = keras.layers.Dense(self.action_dim, activation="softmax")(x)

        return keras.Model(inputs=inputs, outputs=action)
    
    def predict(self, obs):
        return self.model.predict(obs)
    
    def call(self, obs,training=False):
        return self.model(obs,training=training)
    
    def train(self, obs, target):
        self.model.fit(obs,target, epochs=1, verbose=0)

    def trainable_variables(self):
        return self.model.trainable_variables
    
    def update_weights(self,weights):
        self.model.set_weights(weights=weights)

    def get_weights(self):
        return self.model.get_weights()
    
    def save_model(self,name):
        self.model.save(name)
    
    def load_model(self,name):
        return keras.models.load_model(name)
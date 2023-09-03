import tensorflow as tf
from tensorflow import keras


class QNetworkModel:

    def __init__(self,action_dim, dueling=False):
        self.action_dim = action_dim
        
        self.model = self.create_model()
    
    def create_model(self):
        # Network defined by the Deepmind paper
        inputs = keras.layers.Input(shape=(84, 84, 4,))

        # Convolutions on the frames on the screen
        layer1 = keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = keras.layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = keras.layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        layer4 = keras.layers.Flatten()(layer3)

        layer5 = keras.layers.Dense(512, activation="relu")(layer4)
        action = keras.layers.Dense(self.action_dim, activation="relu")(layer5)

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
    

class DuelingQNetworkModel:

    def __init__(self,action_dim):
        self.action_dim = action_dim
        self.model = self.create_dueling_model()
    
    def create_dueling_model(self):
        # Network defined by the Deepmind paper
        inputs = keras.layers.Input(shape=(84, 84, 4,))

        # Convolutions on the frames on the screen
        layer1 = keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = keras.layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = keras.layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        layer4 = keras.layers.Flatten()(layer3)

        layer5 = keras.layers.Dense(512, activation="relu")(layer4)

        #Advantage and Value Streams
        advantage = keras.layers.Dense(self.action_dim, activation="linear")(layer5)
        value = keras.layers.Dense(1, activation="linear")(layer5)

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
    

class PPONetworkModel:

    def __init__(self,action_dim):
        self.action_dim = action_dim
        self.model = self.create_dueling_model()
    
    def create_model(self):
        # Network defined by the Deepmind paper
        inputs = keras.layers.Input(shape=(84, 84, 4,))

        # Convolutions on the frames on the screen
        layer1 = keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = keras.layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = keras.layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        layer4 = keras.layers.Flatten()(layer3)

        layer5 = keras.layers.Dense(512, activation="relu")(layer4)
        action = keras.layers.Dense(self.action_dim, activation="softmax")(layer5)

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
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, Flatten, GRUCell
from tensorflow.keras.models import Model

class OnlineLearner:
    '''Online-learning Multilayer Perceptron'''
    
    def __init__(self, input_dim, hidden_layers = 3, hidden_dim = 32, output_dim = 3, dropout_rate = 0.2, nonlinearity = "relu"):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.nonlinearity = nonlinearity
        
        self.model = None
        
        self.reset_model()
    
    # Define the network architecture
    def reset_model(self):
        i = Input(self.input_dim)
        
        x = i
        
        for _ in range(self.hidden_layers):
            if self.dropout_rate:
                x = Dropout(self.dropout_rate)(x)
            x = Dense(self.hidden_dim, activation = self.nonlinearity)(x)
        
        x = Dense(self.output_dim)(x)
        
        self.model = Model(inputs = i, outputs = x)
    
    def predict(self, X):
        assert(len(X.shape) == 2)   # standard tensorflow shape nonsense
        return self.model.predict(X)
    
    # Choose loss function for training.  Default is MSE.
    def loss_function(self, target, prediction):
        return tf.keras.losses.MeanSquaredError(target, prediction)
    
    # Perform one step of stochastic gradient descent
    def sgd(self, X, T, learning_rate = 0.01, momentum = 0.9):
        assert(len(X.shape) == 2)
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, beta_1 = momentum)
        
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            prediction = self.model(X)
            loss = self.loss_function(T, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    
    def save_weights(self, filepath):
        self.model.save(filepath)
    
    def load_weights(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
    
    def get_weights(self):
        flat_weights = []
        for weight in self.model.get_weights():
            flat_weights.append(weight.flatten())
        return np.concatenate(flat_weights)
    
    def set_weights(self, flat_weights):
        weight_sizes = [self.input_dim] + [self.hidden_dim] * self.hidden_layers + [self.output_dim]
        weights = []
        placeholder = 0

        for i in range(len(weight_sizes) - 1):
            matrix = np.array(flat_weights[placeholder: placeholder + weight_sizes[i] * weight_sizes[i+1]]).reshape(weight_sizes[i], weight_sizes[i + 1])
            weights.append(matrix)
            bias = np.array(flat_weights[placeholder + weight_sizes[i] * weight_sizes[i + 1]: placeholder + weight_sizes[i] * weight_sizes[i + 1] + weight_sizes[i + 1]])
            weights.append(bias)
            placeholder += weight_sizes[i] * weight_sizes[i + 1] + weight_sizes[i + 1]
        
        self.model.set_weights(weights)

class ActorCriticNetwork(OnlineLearner):
    '''Two-headed network for AC algorithms'''

    def reset_model(self):
        i = Input(self.input_dim)
        
        x = i
        
        for _ in range(self.hidden_layers):
            if self.dropout_rate:
                x = Dropout(self.dropout_rate)(x)
            x = Dense(self.hidden_dim, activation = self.nonlinearity)(x)
        
        
        pi = Dense(self.output_dim, activation = "softmax", name = "actor")(x)
        v = Dense(1, name = "critic")(x)
        
        self.model = Model(inputs = i, outputs = [pi, v])
    
    # Override predict method because of shape weirdness with two outputs
    def predict(self, X):
        return [(lambda x : x.squeeze())(x) for x in super().predict(X)]

    # Note that below prediction is [actor prediction, critic prediction], while target is [action_index, reward]
    def loss_function(self, target, prediction, beta = 0.01):
        action_probs = prediction[0]   # these are tensorflow variables!
        value = prediction[1]
        action_index = target[0]
        reward = target[1]

        advantage = reward - value
        actor_loss = - advantage * tf.math.log(action_probs[0, action_index])
        huber = tf.keras.losses.Huber()
        critic_loss = huber(reward, value )
        entropy_loss = tf.reduce_sum((-tf.math.log(action_probs[0] + 1e-9) * action_probs[0]))

        return actor_loss + critic_loss - beta * entropy_loss

class ActorCriticConvNetwork(OnlineLearner):
    '''Two-headed network for AC algorithms'''

    def reset_model(self):
        i = Input(self.input_dim)
        
        x = i
        
        for _ in range(self.hidden_layers):
            x = Conv1D(self.hidden_dim, 3, activation = self.nonlinearity)(x)
        
        x = Flatten()(x)
        x = GRUCell(256)(x)

        pi = Dense(self.output_dim, activation = "softmax", name = "actor")(x)
        v = Dense(1, name = "critic")(x)
        
        self.model = Model(inputs = i, outputs = [pi, v])
    
    # Override predict method because of shape weirdness with two outputs
    def predict(self, X):
        return [(lambda x : x.squeeze())(x) for x in super().predict(X)]

    # Note that below prediction is [actor prediction, critic prediction], while target is [action_index, reward]
    def loss_function(self, target, prediction):
        action_probs = prediction[0]   # these are tensorflow variables!
        value = prediction[1]
        action_index = target[0]
        reward = target[1]

        advantage = reward - value
        actor_loss = - advantage * tf.math.log(action_probs[0, action_index])
        huber = tf.keras.losses.Huber()
        critic_loss = huber(reward, value )

        return actor_loss + critic_loss


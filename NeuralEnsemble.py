import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

# model = NeuralEnsemble(boosting_steps=1000, learning_rate=0.01)
# model.fit(X, y)

class Layer:
    def __init__(self):
        pass
    
    def forward(self, inputs, train=True):
        pass
    
    def backward(self, inputs, grad_outputs):
        pass

class ELU(Layer):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def forward(self, inputs, train=True):
        is_negative = (inputs <= 0)
        is_positive = (inputs > 0)
        negative_part = is_negative * self.alpha * (np.exp(inputs * is_negative) - 1)
        positive_part = inputs * is_positive
        return negative_part + positive_part
    
    def backward(self, inputs, grad_outputs):
        is_negative = (inputs <= 0)
        is_positive = (inputs > 0)
        negative_part = is_negative * self.alpha * np.exp(inputs * is_negative)
        positive_part = 1 * is_positive
        elu_grad = negative_part + positive_part
        return grad_outputs * elu_grad

class BendIdentity(Layer):
    def __init__(self):
        pass

    def forward(self, inputs, train=True):
        return (((inputs**2+1)**(1/2) - 1) / 2) + inputs

class Dense(Layer):
    def __init__(self, input_units, output_units, dropout_rate=0.3, weight=1):
        self.weight = weight
        self.dropout_mask = np.random.binomial(1, (1-dropout_rate), size=(input_units, output_units))
        self.weights = np.ones((input_units, output_units)) * self.weight * self.dropout_mask
        self.biases = np.zeros(output_units)
        
    def forward(self, inputs, train=True):
        return np.dot(inputs, self.weights) + self.biases
    
    def backward(self, inputs, grad_outputs):
        grad_inputs = np.dot(grad_outputs, self.weights.T)
        
        return grad_inputs
        

class MSE():
    def __init__(self):
        pass

    def loss(self, y_true, y_pred):
        return np.mean((y_true-y_pred)**2)

    def grad(self, y_true, y_pred):
        return -(y_true-y_pred)

class NeuralEnsemble():
    def __init__(self, boosting_steps=100, learning_rate=0.1):
        self.boosting_steps = boosting_steps
        self.learning_rate = learning_rate
        self.networks = []
        self.w_values = []
        self.predictions = None
        self.metric = MSE()
    
    def forward(self, network, X, train=True):
        activations = []
        inputs = X

        for layer in network:
            activations.append(layer.forward(inputs, train))
            inputs = activations[-1]
        return activations   
    
    def build_network(self, w):
        layer1 = Dense(2, 50, dropout_rate=0.1, weight=w)
        layer2 = Dense(50, 50, dropout_rate=0.1, weight=w)
        layer3 = Dense(50, 1, dropout_rate=0.1, weight=w)

        network = []
        network.append(layer1)
        network.append(BendIdentity())
        network.append(layer2)
        network.append(BendIdentity())
        network.append(layer3)
        
        return network
    
    def train_step(self, w, X_train, y_train):
        network = self.build_network(w)

        activations = self.forward(network, X_train)
        out = activations[-1]

        loss = self.metric.loss(y_train, out.reshape(-1,))
        return loss
    
    def fit(self, X, y):
        self.predictions = np.zeros(y.shape)
        
        for boosting_step in range(self.boosting_steps):
            
            # Need to update the target outside of the function that we optimize
            residual = y - self.predictions

            # subsample
            X_train, _, y_train, _ = train_test_split(X, residual, test_size=0.1)
            
            # Optimization
            res = minimize(self.train_step, 1, args=(X_train, y_train), method='Powell')
            
            self.w_values.append(res.x)
            network = self.build_network(res.x)
            self.networks.append(network)
            
            activations = self.forward(network, X)
            output = activations[-1]
            
            
            self.predictions += self.learning_rate * output.reshape(-1,)
    
            loss = self.metric.loss(y, self.predictions)
            self.residual = residual
            if boosting_step%10==0:
                print(loss**(1/2))
    
    def predict(self, X):
        predictions = np.zeros((X.shape[0], 1))
        for boosting_step in range(len(self.w_values)):
            network = self.networks[boosting_step]
            activations = self.forward(network, X, train=True)
            boosting_step_predictions = activations[-1]
            predictions += self.learning_rate * boosting_step_predictions
            
        return predictions

import numpy as np 

class Layer:
    def __init__(self):
        pass
    
    def forward(self, inputs, train=True):
        pass

    def backward(self, inputs, grad_outputs):
        pass


class ReLU(Layer):
    def __init__(self):
        pass
    
    def forward(self, inputs, train=True):
        return np.maximum(0, inputs)
    
    def backward(self, inputs, grad_outputs):
        relu_grad = inputs > 0
        return grad_outputs*relu_grad 

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


class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, 
                                        scale = np.sqrt(2/(input_units+output_units)), 
                                        size = (input_units,output_units))
        self.biases = np.zeros(output_units)
        
    def forward(self, inputs, train=True):
        return np.dot(inputs,self.weights) + self.biases
    
    def backward(self, inputs, grad_outputs):
        grad_inputs = np.dot(grad_outputs, self.weights.T)
        grad_weights = np.dot(inputs.T, grad_outputs)
        grad_biases = grad_outputs.mean(axis=0)*inputs.shape[0]

        self.weights -= self.learning_rate * grad_weights
        self.biases -= self.learning_rate * grad_biases
        
        return grad_inputs

class Conv2D(Layer):
    def __init__(self, in_channels, num_filters, kernel_size, strides=1, padding=0, learning_rate=0.1):
        
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.filters = (np.random.random(size=(kernel_size, kernel_size, in_channels, num_filters)) * 2 - 1) / np.sqrt(1.0 / kernel_size*num_filters)
        self.biases = np.zeros(num_filters)
        self.strides = strides
        self.padding = padding
        self.learning_rate = learning_rate
        
    def zero_pad(self, X, pad):
        return np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
                
    def forward(self, inputs, train=True):
        """
        inputs: B x H x W x C  ---- (B)atch size x (H)eight x (W)idth x (C)hannel size
        outputs: B x O1 x O2 x F  ---- (B)atch size x (O1)Height x (O2)Width x (F)ilter size
        O1 = (W - K1 + 2*P ) / S + 1
        O2 = (L - K2 + 2*P ) / S + 1
        F: number of filters
        """
        (batch_size, H_in, W_in, C_prev) = inputs.shape
        (kernel_size, kernel_size, C_in, C_out) = self.filters.shape
        
        H_out = int((H_in - kernel_size + 2*self.padding)/self.strides) + 1
        W_out = int((W_in - kernel_size + 2*self.padding)/self.strides) + 1

        inputs_padded = self.zero_pad(inputs, self.padding)

        self.outputs = np.zeros((batch_size, H_out, W_out, C_out)) 

        for h in range(H_out):
            for w in range(W_out):
                h1 = h * self.strides
                h2 = h * self.strides + self.kernel_size
                w1 = w * self.strides
                w2 = w * self.strides + self.kernel_size
                self.outputs[:, h, w, :] = np.sum(np.expand_dims(self.filters, 0) * 
                                             np.expand_dims(inputs_padded[:, h1:h2, w1:w2, :], -1), axis=(1, 2, 3)) + self.biases

                
        return self.outputs
    
    def backward(self, inputs, grad_outputs):
        '''
        Backpropagation through a convolutional layer. 
        '''
        (batch_size, H_in, W_in, C_input) = inputs.shape
        (batch_size, H_out, W_out, C_out) = grad_outputs.shape
        (kernel_size, kernel_size, C_in, C_out) = self.filters.shape
        grad_inputs = np.zeros(inputs.shape) 
        grad_filters = np.zeros(self.filters.shape)
        grad_biases = np.zeros((C_out,1))
        
        for h in range(H_out):
            for w in range(W_out):
                h1 = h * self.strides
                h2 = h * self.strides + self.kernel_size
                w1 = w * self.strides
                w2 = w * self.strides + self.kernel_size
                grad_outputs_reshaped = np.expand_dims(np.expand_dims(grad_outputs[:, h, w, :], 1), 2)
                grad_filters += np.sum(np.expand_dims(grad_outputs_reshaped, 3) * np.expand_dims(inputs[:, h1:h2, w1:w2, :], 4), axis=0)
                grad_inputs[:, h1:h2, w1:w2, :] += np.sum(np.expand_dims(grad_outputs_reshaped, 3) * np.expand_dims(self.filters, 0), axis=4)

        grad_biases = grad_outputs.mean(axis=(0,1,2))*inputs.shape[0]
        
        grad_biases = np.squeeze(grad_biases)
        self.filters -= self.learning_rate * grad_filters
        self.biases -= self.learning_rate * grad_biases
        return grad_inputs


class Flatten(Layer):

    def __init__(self):
        self.shape = None
    
    def forward(self, inputs, train=True):
        self.shape = inputs.shape
        return inputs.reshape((self.shape[0], np.prod(self.shape[1:])))
    
    def backward(self, inputs, grad_outputs):
        grad_outputs = grad_outputs.reshape(self.shape)
        return grad_outputs


class MaxPool2D(Layer):
    
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self, inputs, train=True):
        (batch_size, H_in, W_in, C_input) = inputs.shape

        H_out = int(1 + (H_in - self.kernel_size) / self.stride)
        W_out = int(1 + (W_in - self.kernel_size) / self.stride)

        outputs = np.zeros((batch_size, H_out, W_out, C_input)) 

        for h in range(H_out):
            for w in range(W_out):
                h1 = h * self.stride
                h2 = h * self.stride + self.kernel_size
                w1 = w * self.stride
                w2 = w * self.stride + self.kernel_size
                window = inputs[:, h1:h2, w1:w2, :]
                window = np.reshape(window, (-1, 1))
                outputs[:, h, w, :] = np.max(window.reshape((batch_size, self.kernel_size*self.kernel_size, C_input)), axis=(1, 2)).reshape(-1,1)

        return outputs
    
    def backward(self, inputs, grad_outputs):

        (batch_size, H_in, W_in, C_input) = inputs.shape
        (batch_size, H_out, W_out, C_out) = grad_outputs.shape

        grad_inputs = np.zeros(inputs.shape)

        for h in range(H_out):
            for w in range(W_out):
                h1 = h * self.stride
                h2 = h * self.stride + self.kernel_size
                w1 = w * self.stride
                w2 = w * self.stride + self.kernel_size

                window = inputs[:, h1:h2, w1:w2, :]
                window = np.reshape(window, (batch_size, self.kernel_size*self.kernel_size, C_input))

                mask = np.zeros_like(window)
                mask[:, np.argmax(window, axis=1), :] = 1
                mask = np.reshape(mask, (batch_size, self.kernel_size, self.kernel_size, C_input))
                grad_outputs_reshaped = np.expand_dims(np.expand_dims(grad_outputs[:, h, w, :], 1), 2)
                
                grad_inputs[:, h1:h2, w1:w2, :] += mask * grad_outputs_reshaped

        return grad_inputs

class AveragePool2D(Layer):
    
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self, inputs, train=True):
        (batch_size, H_in, W_in, C_input) = inputs.shape

        H_out = int(1 + (H_in - self.kernel_size) / self.stride)
        W_out = int(1 + (W_in - self.kernel_size) / self.stride)

        outputs = np.zeros((batch_size, H_out, W_out, C_input)) 

        for h in range(H_out):
            for w in range(W_out):
                h1 = h * self.stride
                h2 = h * self.stride + self.kernel_size
                w1 = w * self.stride
                w2 = w * self.stride + self.kernel_size
                window = inputs[:, h1:h2, w1:w2, :]
                window = np.reshape(window, (-1, 1))
                outputs[:, h, w, :] = np.mean(window.reshape((batch_size, self.kernel_size*self.kernel_size, C_input)), axis=(1, 2)).reshape(-1,1)

        return outputs
    
    def backward(self, inputs, grad_outputs):

        (batch_size, H_in, W_in, C_input) = inputs.shape
        (batch_size, H_out, W_out, C_out) = grad_outputs.shape

        grad_inputs = np.zeros(inputs.shape)

        for h in range(H_out):
            for w in range(W_out):
                h1 = h * self.stride
                h2 = h * self.stride + self.kernel_size
                w1 = w * self.stride
                w2 = w * self.stride + self.kernel_size

                window = inputs[:, h1:h2, w1:w2, :]
                mean_value = np.ones(window.shape)
                grad_outputs_reshaped = np.expand_dims(np.expand_dims(grad_outputs[:, h, w, :], 1), 2)
                mean_value *= grad_outputs_reshaped/(self.kernel_size * self.kernel_size)

                grad_inputs[:, h1:h2, w1:w2, :] += mean_value

        return grad_inputs

class BatchNorm(Layer):
    def __init__(self, input_units, gamma=1, beta=0, momentum=0.99, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.beta = beta
        self.mu = np.zeros(input_units)
        self.var = np.ones(input_units)
        self.batch_mu = None
        self.batch_var = None
        self.momentum = momentum
        self.outputs = None
        self.inputs_norm = None

    def forward(self, inputs, train=True):

        if train:
            if len(inputs.shape) == 4:
                self.batch_mu = np.mean(inputs, axis=(0,1,2))
                self.batch_var = np.var(inputs, axis=(0,1,2))
            else:
                self.batch_mu = np.mean(inputs, axis=0)
                self.batch_var = np.var(inputs, axis=0)

            self.mu = self.momentum * self.mu + (1 - self.momentum) * self.batch_mu
            self.var = self.momentum * self.var + (1 - self.momentum) * self.batch_var

            self.inputs_norm = (inputs - self.batch_mu) / np.sqrt(self.batch_var + 1e-8)
            self.outputs = self.gamma * self.inputs_norm + self.beta
        
        else:
            self.inputs_norm = (inputs - self.mu) / np.sqrt(self.var + 1e-8)
            self.outputs = self.gamma * self.inputs_norm + self.beta            

        return self.outputs

    def backward(self, inputs, grad_outputs):

        if len(inputs.shape) == 4:
            (batch_size, _, _, _) = inputs.shape
        else:
            (batch_size, _) = inputs.shape

        inputs_mu = inputs - self.mu
        one_over_var = 1. / np.sqrt(self.var + 1e-8)

        grad_inputs_norm = grad_outputs * self.gamma
        grad_var = np.sum(grad_inputs_norm * inputs_mu, axis=0) * -.5 * one_over_var**3
        grad_mu = np.sum(grad_inputs_norm * -one_over_var, axis=0) + grad_var * np.mean(-2. * inputs_mu, axis=0)

        grad_inputs = (grad_inputs_norm * one_over_var) + (grad_var * 2 * inputs_mu / batch_size) + (grad_mu / batch_size)
        grad_gamma = np.sum(grad_outputs * self.inputs_norm, axis=0)
        grad_beta = np.sum(grad_outputs, axis=0)

        self.gamma -= self.learning_rate * grad_gamma
        self.beta -= self.learning_rate * grad_beta

        return grad_inputs

class Dropout(Layer):
    def __init__(self, rate=0.1, learning_rate=0.1):
        self.rate = rate
        self.outputs = None
        self.learnin_rate = learning_rate
        self.dropout = None
    
    def forward(self, inputs, train=True):
        if train:
            self.dropout = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        else:
            self.dropout = np.ones(inputs.shape)

        self.outputs = inputs * self.dropout
        return self.outputs

    def backward(self, inputs, grad_outputs):
        return grad_outputs * self.dropout
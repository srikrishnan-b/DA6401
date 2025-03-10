import numpy as np
from utils import sigmoid, softmax, relu, tanh


# Layer class for the neural network
class Layer:
    def __init__(self, name, input_size, n_units, activation=None):
        self.n = name
        self.input_size = input_size
        self.n_units = n_units
        self.weights = None
        self.bias = None
        self.g = activation
        self.grad_w = None
        self.grad_b = None
        self.a = None  # Preactivatiuon
        self.h = None  # Activation
        self.temp_weights = None
        self.temp_bias = None  # Lookahead parameters for NAG

    def __call__(self, input):
        hidden = input @ self.weights.T + self.bias
        self.preact = hidden
        self.act = self.activation(hidden)
        self.squared_weights = self.weights**2
        # self.squared_bias = self.bias**2
        return self.act

    def activation(self, z):
        if self.g == "sigmoid":
            return sigmoid(z)
        elif self.g == "softmax":
            return softmax(z)
        elif self.g == "ReLU":
            return relu(z)
        elif self.g == "tanh":
            return tanh(z)
        elif self.g == "identity":
            return z


# Multilayer perceptron class
class NeuralNet:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        n_hidden,
        loss: str,
        activation=["sigmoid", "softmax"],
        weight_init="random",
        weight_decay=0,
    ):
        self.n_layers = n_hidden + 1
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_activation = activation[0]
        self.output_activation = activation[1]
        self.layers = self._network()
        self.alpha = weight_decay  # L2 regularization parameter
        self.squared_weights_sum = 0
        self.squared_bias_sum = 0
        self.initialize_weights(init_type=weight_init)
        self.loss_fn = loss
        self.loss = 0
        self.acc = None

    def random_weights(self):
        for layer in self.layers:
            layer.weights = np.random.randn(layer.n_units, layer.input_size)
            layer.bias = np.random.randn(layer.n_units)

    def xavier_weights(self):

        for layer in self.layers:
            input_n = layer.input_size
            output_n = layer.n_units
            low, high = -np.sqrt(6 / (input_n + output_n)), np.sqrt(
                6 / (input_n + output_n)
            )
            layer.weights = np.random.uniform(
                low, high, (layer.n_units, layer.input_size)
            )
            layer.bias = np.random.uniform(low, high, (layer.n_units))

    def initialize_weights(self, init_type="random"):
        if init_type == "random":
            self.random_weights()
        elif init_type == "xavier":
            self.xavier_weights()

    def _network(self):
        layers = []
        for i in range(self.n_layers):
            if i == 0:
                layers.append(
                    Layer(
                        i,
                        self.input_size,
                        self.hidden_size,
                        activation=self.hidden_activation,
                    )
                )
            elif i < self.n_layers - 1:
                layers.append(
                    Layer(
                        i,
                        self.hidden_size,
                        self.hidden_size,
                        activation=self.hidden_activation,
                    )
                )
            else:
                layers.append(
                    Layer(
                        i,
                        self.hidden_size,
                        self.output_size,
                        activation=self.output_activation,
                    )
                )

        return layers

    def get_weights(self):
        return [layer.weights for layer in self.layers]

    def get_biases(self):
        return [layer.bias for layer in self.layers]

    def __call__(self, x, y):
        for layer in self.layers:
            x = layer(x)
            self.squared_weights_sum += np.sum(layer.squared_weights)
            # self.squared_bias_sum += np.sum(layer.squared_bias)

        self.loss = (
            self.loss_fn(x, y) + self.alpha * (self.squared_weights_sum) / 2
        )  # + self.squared_bias_sum) / 2    # L2 regularization loss term
        self.squared_weights_sum = 0
        self.acc = self.accuracy(x, y)
        return x, self.loss, self.acc

    def predict(self, x, y=None):
        for layer in self.layers:
            x = layer(x)
            self.squared_weights_sum += np.sum(layer.squared_weights)

        self.loss = (
            self.loss_fn(x, y) + self.alpha * (self.squared_weights_sum) / 2
        )  # + self.squared_bias_sum) / 2
        self.squared_weights_sum = 0
        if y is not None:
            self.acc = self.accuracy(x, y)
        self.acc = self.accuracy(x, y)
        return x, self.loss, self.acc

    def accuracy(self, y_pred, y_true):
        correct = np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)
        return np.mean(correct)

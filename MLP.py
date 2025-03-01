import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist


# Load the Fashion-MNIST data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


# Getting the unique labels and one entry from each
labels = set(y_train)
sample = {ind: np.where(y_train == ind)[0][0] for ind in labels}

# Plotting one image from each label
for i in range(10):
    plt.subplot(4, 3, i + 1)
    plt.imshow(x_train[sample[i]], cmap="gray")
    plt.title(f"label={i}", fontsize=8)

plt.show()


def one_hot_encode(y):
    vec = np.zeros((10,))
    vec[y] = 1
    return vec


# reshaping the data
x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_train = x_train / x_train.sum()
x_test = x_test.reshape(x_test.shape[0], 28 * 28)
x_test = x_test / x_test.sum()
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

# Converting the labels to one-hot encoding
y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Activation functions
eps = 1e-5


def sigmoid(x):
    return 1 / (1 + np.exp(-x) + eps)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


# Layer class
class Layer:
    def __init__(self, input_size, n_units, activation=None):
        self.weights = np.random.randn(input_size, n_units)
        self.bias = np.random.randn(n_units)
        self.g = activation

    def __call__(self, input):
        print(self.weights.shape)
        print(input.shape)
        hidden = np.matmul(input, self.weights) + self.bias
        hidden = self.activation(hidden)
        return hidden

    def activation(self, z):
        if self.g == "sigmoid":
            return sigmoid(z)
        elif self.g == "softmax":
            return softmax(z)


# Neural Network class
class NeuralNet:
    def __init__(self, input_size, n_units: list):
        self.n_layers = len(n_units)
        self.n_units = n_units
        self.input_size = input_size
        self.layers = self._network()

    def _network(self):
        layers = []
        for i in range(self.n_layers):
            if i == 0:
                layers.append(
                    Layer(self.input_size, self.n_units[i], activation="sigmoid")
                )
            elif i < self.n_layers - 1:
                layers.append(
                    Layer(self.n_units[i - 1], self.n_units[i], activation="sigmoid")
                )
            else:
                layers.append(
                    Layer(self.n_units[i - 1], self.n_units[i], activation="softmax")
                )

        return layers

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            print(i)
            x = layer(x)

        return x


def batch_data(x, y, batch_size):
    for i in range(0, len(x), batch_size):
        yield x[i : i + batch_size], y[i : i + batch_size]


MLP = NeuralNet(x_train.shape[1], [128, 64, 32, 10])

for x, y in batch_data(x_train, y_train, 32):
    out = MLP(x)

print(out.shape)

import numpy as np

eps = 1e-10


# ===============================================================================
# Activation functions
# ===============================================================================
def sigmoid(x):
    x = np.clip(x, -500, 500)  # To avoid overflow
    return 1 / (1 + np.exp(-x) + eps)


def softmax(x):
    max_x = np.max(x, axis=1, keepdims=True)
    x = x - max_x
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


# ===============================================================================


# ===============================================================================
# Loss function
# ===============================================================================
class Loss:
    def __init__(self, loss_fn="cross_entropy"):
        self.loss_fn = loss_fn

    def __call__(self, y_true, y_pred):
        if self.loss_fn == "cross_entropy":
            return np.mean(self.cross_entropy(y_true, y_pred))
        elif self.loss_fn == "mse":
            return np.mean(self.mse(y_true, y_pred))

    def cross_entropy(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # To avoid log(0) and log(1)
        return -np.log(np.sum(y_true * y_pred, axis=1))

    def mse(self, y_true, y_pred):
        return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))


# ===============================================================================
def one_hot_encode(y):
    one_hot_vectors = []
    for i in y:
        vec = np.zeros((10,))
        vec[i] = 1
        one_hot_vectors.append(vec)
    return np.array(one_hot_vectors)


# ===============================================================================


# ===============================================================================
def batch_data(x, y, batch_size):
    batches = []
    for i in range(0, x.shape[0], batch_size):
        batches.append((x[i : i + batch_size], y[i : i + batch_size]))
    return batches


# ===============================================================================

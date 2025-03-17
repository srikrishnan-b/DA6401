import numpy as np
import copy


class GradientDescent:
    def __init__(
        self,
        optimizer="sgd",
        lr=0.01,
        clipping_threshold=1,
        momentum=0.9,
        beta=0.9,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-10,
    ):
        self.optimizer = optimizer
        self.lr = lr
        self.threshold = clipping_threshold
        self.momentum = momentum
        self.beta = beta
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def optimize(self, model, x, y, i=0):
        if self.optimizer == "sgd":
            return self.SGD(model, x, y, i)
        elif self.optimizer == "momentum":
            return self.momentum_gd(model, x, y, i)
        elif self.optimizer == "nag":
            return self.NAG(model, x, y, i)
        elif self.optimizer == "rmsprop":
            return self.RMSprop(model, x, y, i)
        elif self.optimizer == "adam":
            return self.Adam(model, x, y, i)
        elif self.optimizer == "nadam":
            return self.Nadam(model, x, y, i)

    # =========================================================================================
    # Backpropagation for all optimizers
    # =========================================================================================
    def backprop(self, model, x, y, output, i):
        # For computing gradients and updating weights
        grad_w, grad_b = self.compute_grads(model, x, y, output)
        self.clip_all_gradients(model, grad_w, grad_b)
        self.update_weights(model, grad_w, grad_b, i)

    # ================================END OF SECTION===========================================

    # =========================================================================================
    # Gradient computation for all layers
    # =========================================================================================
    def compute_grads(self, model, x, y, output):
        batch_size = x.shape[0]
        grad_a = {}
        grad_h = {}
        grad_w = {}
        grad_b = {}
        for layer in model.layers[::-1]:
            # Gradients wrt to the output layer, preactivation, activation, weights and biases
            if layer.n == model.n_layers - 1:

                grad_a[layer.n] = self.grad_output(y, output, model.loss_fn).reshape(
                    batch_size, model.output_size, 1
                )
                grad_w[layer.n] = (
                    np.sum(
                        self.grad_weights(
                            grad_a[layer.n],
                            model.layers[layer.n - 1].h.reshape(
                                batch_size, model.hidden_size, 1
                            ),
                        ),
                        axis=0,
                    )
                    + model.alpha * layer.weights
                )
                grad_b[layer.n] = np.sum(self.grad_bias(grad_a[layer.n]), axis=0)

            # Gradients wrt to the hidden layers, preactivation, activation, weights and biases
            else:
                grad_h[layer.n] = self.grad_hidden(
                    model.layers[layer.n + 1].weights, grad_a[layer.n + 1]
                )
                grad_a[layer.n] = self.grad_preact(
                    grad_h[layer.n],
                    model.layers[layer.n].a.reshape(batch_size, model.hidden_size, 1),
                    model.hidden_activation,
                )

                # Gradients wrt to the input layer, preactivation, activation, weights and biases
                if layer.n == 0:
                    grad_w[layer.n] = (
                        np.sum(
                            self.grad_weights(
                                grad_a[layer.n],
                                x.reshape(batch_size, model.input_size, 1),
                            ),
                            axis=0,
                        )
                        + model.alpha * layer.weights
                    )
                else:
                    grad_w[layer.n] = (
                        np.sum(
                            self.grad_weights(
                                grad_a[layer.n],
                                model.layers[layer.n - 1].h.reshape(
                                    batch_size, model.hidden_size, 1
                                ),
                            ),
                            axis=0,
                        )
                        + model.alpha * layer.weights
                    )

                grad_b[layer.n] = np.sum(self.grad_bias(grad_a[layer.n]), axis=0)

        return grad_w, grad_b

    # ================================END OF SECTION===========================================

    # =========================================================================================
    # Weights updates for all optimizers
    # =========================================================================================
    def update_weights(self, model, grad_w, grad_b, i):
        if self.optimizer == "sgd":
            self.update_weights_SGD(model, grad_w, grad_b)

        elif self.optimizer == "momentum":
            self.update_weights_momentum(model, grad_w, grad_b)

        elif self.optimizer == "nag":
            self.update_weights_nag(model, grad_w, grad_b)

        elif self.optimizer == "rmsprop":
            self.update_weights_rmsprop(model, grad_w, grad_b)

        elif self.optimizer == "adam":
            self.update_weights_adam(model, grad_w, grad_b, i)

        elif self.optimizer == "nadam":
            self.update_weights_nadam(model, grad_w, grad_b, i)

    # ================================END OF SECTION===========================================

    # =========================================================================================
    # Optimizers
    # =========================================================================================
    def SGD(self, model, x, y, i=0):
        output, loss, acc = model(x, y)  # Forward pass
        self.backprop(model, x, y, output, i)  # Backpropagation
        return loss, acc

    def momentum_gd(self, model, x, y, i=0):
        if i == 0:
            self.gradw_his = {}
            self.gradb_his = {}
        output, loss, acc = model(x, y)  # Forward pass
        self.backprop(model, x, y, output, i)  # Backpropagation
        return loss, acc

    def NAG(self, model, x, y, i=0):
        if i == 0:
            self.gradw_his = {}
            self.gradb_his = {}
            for layer in model.layers:
                self.gradw_his[layer.n] = np.zeros_like(layer.weights)
                self.gradb_his[layer.n] = np.zeros_like(layer.bias.reshape(-1, 1))
        for layer in model.layers:
            layer.temp_weights = copy.deepcopy(layer.weights)
            layer.temp_bias = copy.deepcopy(layer.bias)
            layer.weights = layer.weights - (self.beta * self.gradw_his[layer.n])
            layer.bias = layer.bias - (self.beta * self.gradb_his[layer.n].squeeze())

        output, loss, acc = model(x, y)  # Forward pass
        self.backprop(model, x, y, output, i)  # Backpropagation
        return loss, acc

    def RMSprop(self, model, x, y, i=0):
        if i == 0:
            self.gradw_his = {}
            self.gradb_his = {}
        output, loss, acc = model(x, y)  # Forward pass
        self.backprop(model, x, y, output, i)  # Backpropagation
        return loss, acc

    def Adam(self, model, x, y, i=0):
        if i == 0:
            self.gradw_his = {}
            self.gradb_his = {}
            self.gradw_m = {}
            self.gradb_m = {}

        output, loss, acc = model(x, y)  # Forward pass
        self.backprop(model, x, y, output, i)  # Backpropagation
        return loss, acc

    def Nadam(self, model, x, y, i=0):
        if i == 0:
            self.gradw_his = {}
            self.gradb_his = {}
            self.gradw_m = {}
            self.gradb_m = {}

        output, loss, acc = model(x, y)  # Forward pass
        self.backprop(model, x, y, output, i)  # Backpropagation
        return loss, acc

    # ================================END OF SECTION===========================================

    # =========================================================================================
    # Update rules for all optimizers
    # =========================================================================================
    def update_weights_SGD(self, model, grad_w, grad_b):
        # Standard update rule for SGD
        for layer in model.layers:
            layer.weights = layer.weights - self.lr * grad_w[layer.n]
            layer.bias = layer.bias - self.lr * grad_b[layer.n].squeeze()

    def update_weights_momentum(self, model, grad_w, grad_b):
        # Initiate history for Momentum
        if self.gradw_his == {} and self.gradb_his == {}:
            for layer in model.layers:
                self.gradw_his[layer.n] = np.zeros_like(layer.weights)
                self.gradb_his[layer.n] = np.zeros_like(layer.bias).reshape(-1, 1)

        # Computing current history for Momentum
        for layer in model.layers:
            self.gradw_his[layer.n] = (
                self.momentum * self.gradw_his[layer.n]
            ) + self.lr * grad_w[layer.n]

            self.gradb_his[layer.n] = (
                self.momentum * self.gradb_his[layer.n]
            ) + self.lr * grad_b[layer.n]

            # Update weights and biases using Momentum
            layer.weights = layer.weights - self.gradw_his[layer.n]
            layer.bias = layer.bias - self.gradb_his[layer.n].squeeze()

    def update_weights_nag(self, model, grad_w, grad_b):
        # Computing current history for NAG
        for layer in model.layers:
            self.gradw_his[layer.n] = (
                self.beta * self.gradw_his[layer.n]
            ) + self.lr * grad_w[layer.n]

            self.gradb_his[layer.n] = (
                self.beta * self.gradb_his[layer.n]
            ) + self.lr * grad_b[layer.n]

            layer.weights = copy.deepcopy(layer.temp_weights)
            layer.bias = copy.deepcopy(layer.temp_bias)
            layer.weights = layer.weights - self.gradw_his[layer.n]
            layer.bias = layer.bias - self.gradb_his[layer.n].squeeze()

    def update_weights_rmsprop(self, model, grad_w, grad_b):
        # Initiating history
        if self.gradw_his == {} and self.gradb_his == {}:
            for layer in model.layers:
                self.gradw_his[layer.n] = np.zeros_like(layer.weights)
                self.gradb_his[layer.n] = np.zeros_like(layer.bias).reshape(-1, 1)

        # Computing current history for RMSprop
        for layer in model.layers:
            self.gradw_his[layer.n] = (
                self.beta * self.gradw_his[layer.n]
                + (1 - self.beta) * grad_w[layer.n] ** 2
            )

            self.gradb_his[layer.n] = (
                self.beta * self.gradb_his[layer.n]
                + (1 - self.beta) * grad_b[layer.n] ** 2
            )

            # Update weights and biases using history
            layer.weights = (
                layer.weights
                - (self.lr / np.sqrt(self.gradw_his[layer.n] + self.epsilon))
                * grad_w[layer.n]
            )

            layer.bias = (
                layer.bias
                - (self.lr / np.sqrt(self.gradb_his[layer.n].squeeze() + self.epsilon))
                * grad_b[layer.n].squeeze()
            )

    def update_weights_adam(self, model, grad_w, grad_b, i):
        # Initiating history
        if self.gradw_m == {} and self.gradb_m == {}:
            for layer in model.layers:
                self.gradw_m[layer.n] = np.zeros_like(layer.weights)
                self.gradb_m[layer.n] = np.zeros_like(layer.bias).reshape(-1, 1)
                self.gradw_his[layer.n] = np.zeros_like(layer.weights)
                self.gradb_his[layer.n] = np.zeros_like(layer.bias).reshape(-1, 1)

        # Computing current history and momentum for Adam
        for layer in model.layers:
            self.gradw_m[layer.n] = (
                self.beta1 * self.gradw_m[layer.n] + (1 - self.beta1) * grad_w[layer.n]
            )

            self.gradw_his[layer.n] = (
                self.beta2 * self.gradw_his[layer.n]
                + (1 - self.beta2) * grad_w[layer.n] ** 2
            )

            gradw_mhat = self.gradw_m[layer.n] / (1 - np.power(self.beta1, i + 1))
            gradw_vhat = self.gradw_his[layer.n] / (1 - np.power(self.beta2, i + 1))

            self.gradb_m[layer.n] = (
                self.beta1 * self.gradb_m[layer.n] + (1 - self.beta1) * grad_b[layer.n]
            )
            self.gradb_his[layer.n] = (
                self.beta2 * self.gradb_his[layer.n]
                + (1 - self.beta2) * grad_b[layer.n] ** 2
            )

            gradb_mhat = self.gradb_m[layer.n] / ((1 - np.power(self.beta1, i + 1)))
            gradb_vhat = self.gradb_his[layer.n] / (1 - np.power(self.beta2, i + 1))

            # Update weights and biases using new history and momentum
            layer.weights = layer.weights - (self.lr * gradw_mhat) / (
                np.sqrt(gradw_vhat) + self.epsilon
            )
            layer.bias = layer.bias - (self.lr * gradb_mhat.squeeze()) / (
                np.sqrt(gradb_vhat.squeeze()) + self.epsilon
            )

    def update_weights_nadam(self, model, grad_w, grad_b, i):
        if self.gradw_m == {} and self.gradb_m == {}:
            for layer in model.layers:
                self.gradw_m[layer.n] = np.zeros_like(layer.weights)
                self.gradb_m[layer.n] = np.zeros_like(layer.bias).reshape(-1, 1)
                self.gradw_his[layer.n] = np.zeros_like(layer.weights)
                self.gradb_his[layer.n] = np.zeros_like(layer.bias).reshape(-1, 1)

        for layer in model.layers:
            self.gradw_m[layer.n] = (
                self.beta1 * self.gradw_m[layer.n] + (1 - self.beta1) * grad_w[layer.n]
            )
            self.gradw_his[layer.n] = (
                self.beta2 * self.gradw_his[layer.n]
                + (1 - self.beta2) * grad_w[layer.n] ** 2
            )

            gradw_mhat = self.gradw_m[layer.n] / (1 - np.power(self.beta1, i + 1))
            gradw_vhat = self.gradw_his[layer.n] / (1 - np.power(self.beta2, i + 1))

            self.gradb_m[layer.n] = (
                self.beta1 * self.gradb_m[layer.n] + (1 - self.beta1) * grad_b[layer.n]
            )
            self.gradb_his[layer.n] = (
                self.beta2 * self.gradb_his[layer.n]
                + (1 - self.beta2) * grad_b[layer.n] ** 2
            )

            gradb_mhat = self.gradb_m[layer.n] / ((1 - np.power(self.beta1, i + 1)))
            gradb_vhat = self.gradb_his[layer.n] / (1 - np.power(self.beta2, i + 1))

            layer.weights = layer.weights - (
                self.lr
                * (
                    (self.beta1 * gradw_mhat)
                    + ((1 - self.beta1) * grad_w[layer.n])
                    / (1 - np.power(self.beta1, i + 1))
                )
            ) / (np.sqrt(gradw_vhat) + self.epsilon)
            layer.bias = layer.bias - (
                self.lr
                * (
                    (self.beta1 * gradb_mhat.squeeze())
                    + ((1 - self.beta1) * grad_b[layer.n]).squeeze()
                    / (1 - np.power(self.beta1, i + 1))
                )
            ) / (np.sqrt(gradb_vhat.squeeze()) + self.epsilon)

    # ================================END OF SECTION===========================================

    # =========================================================================================
    # Definition of gradients of loss w.r.t ouput, hidden, preactivation, weights and biases
    # =========================================================================================
    def grad_yhat(self, y_true, y_pred, loss="cross_entropy"):
        if loss == "cross_entropy":
            y_grad = np.sum(y_true * y_pred, axis=1)
            return -(y_true / y_grad[:, np.newaxis])

        elif loss == "mse":
            return y_pred - y_true

    def grad_output(self, y_true, y_pred, loss_fn):
        if loss_fn.loss_fn == "cross_entropy":
            return -(y_true - y_pred)
        elif loss_fn.loss_fn == "mse":
            S = np.sum((y_pred - y_true) * y_pred, axis=1, keepdims=True)  # Jacobian
            grad_pre_final = y_pred * ((y_pred - y_true) - S)

            return grad_pre_final

    def grad_hidden(self, W, grad_next_preact):
        return W.T @ grad_next_preact

    def grad_preact(self, grad_act, act, activation):
        if activation == "sigmoid":  # for sigmoid

            return grad_act * act * (1 - act)
        elif activation == "ReLU":  # for relu
            return grad_act * (act > 0)
        elif activation == "tanh":  # for tanh
            return grad_act * (1 - act**2)
        elif activation == "identity":
            return grad_act

    def grad_weights(self, grad_preact, input):
        return grad_preact @ input.transpose(0, 2, 1)

    def grad_bias(self, grad_preact):
        return grad_preact

    # ================================END OF SECTION===========================================

    # =========================================================================================
    # Clipping gradients, checking for NaNs
    # ================================================================================
    def clip_gradients(self, gradients):
        return np.clip(gradients, -self.threshold, self.threshold)

    def clip_all_gradients(self, model, grad_w, grad_b):
        for layer in model.layers:
            grad_w[layer.n] = self.clip_gradients(grad_w[layer.n])
            grad_b[layer.n] = self.clip_gradients(grad_b[layer.n])

    def check_for_nans(self, array):
        if np.isnan(array).any():
            raise ValueError("NaN detected in array")

    # ================================END OF SECTION===========================================

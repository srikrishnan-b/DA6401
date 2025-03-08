import numpy as np
import copy


class GradientDescent:
    def __init__(
        self,
        optimizer="sgd",
        lr=0.01,
        clipping_threshold=1e5,
        beta=0.9,
        beta_2=0.999,
        eps=1e-10,
    ):
        self.optimizer = optimizer
        self.lr = lr
        self.threshold = clipping_threshold
        self.eps = eps
        self.beta = beta
        self.beta_2 = beta_2

    def optimize(self, model, x, y, i=0):
        if self.optimizer == "sgd":
            pass
        elif self.optimizer == "momentum":
            pass
        elif self.optimizer == "nag":
            pass
        elif self.optimizer == "rmsprop":
            pass
        elif self.optimizer == "adam":
            pass
        elif self.optimizer == "nadam":
            pass

    def SGD(self, model, x, y, i=0):

        output, loss, acc = model(x, y)  # Forward pass
        self.backprop(model, x, y, output, i)  # Backpropagation
        return loss, acc

    def momentum(self, model, x, y, i=0):
        self.gradw_his = {}
        self.gradb_his = {}
        output, loss, acc = model(x, y)  # Forward pass
        self.backprop(model, x, y, output, i)  # Backpropagation
        return loss, acc

    def NAG(self, model, x, y, i=0):
        pass

    def RMSprop(self, model, x, y, i=0):
        self.gradw_his = {}
        self.gradb_his = {}
        output, loss, acc = model(x, y)  # Forward pass
        self.backprop(model, x, y, output, i)  # Backpropagation
        return loss, acc

    def Adam(self, model, x, y, i=0):
        self.gradw_his = {}
        self.gradb_his = {}
        self.gradw_m = {}
        self.gradb_m = {}

        output, loss, acc = model(x, y)  # Forward pass
        self.backprop(model, x, y, output, i)  # Backpropagation
        return loss, acc

    def Nadam(self, model, x, y, i=0):
        self.gradw_his = {}
        self.gradb_his = {}
        self.gradw_m = {}
        self.gradb_m = {}

        output, loss, acc = model(x, y)  # Forward pass
        self.backprop(model, x, y, output, i)  # Backpropagation
        return loss, acc

    def gradient_descent(self, model, x, y, i=0):
        if self.optimizer in [
            "sgd",
            "momentum",
            "rmsprop",
            "adam",
            "nadam",
        ]:  # Forward pass and backprop for SGD and Momentum
            output, loss, acc = model(x, y)
            self.backprop(model, x, y, output, i)

        elif (
            self.optimizer == "nag"
        ):  # Computing gradient at lookahead point before update for NAG
            for layer in model.layers:
                self.gradw_his[layer.n] = [np.zeros_like(layer.weights)]
                self.gradb_his[layer.n] = [np.zeros_like(layer.bias.reshape(-1, 1))]
                # print(layer.n)
                # print("w his, w", self.gradw_his[layer.n][-1].shape, layer.weights.shape)
                # print("b his, b", self.gradb_his[layer.n][-1].shape, layer.bias.shape)
                layer.temp_weights = copy.deepcopy(layer.weights)
                layer.temp_bias = copy.deepcopy(layer.bias)
                layer.weights = layer.weights - self.beta * self.gradw_his[layer.n][-1]
                layer.bias = (
                    layer.bias - self.beta * self.gradb_his[layer.n][-1].squeeze()
                )
                # print("b after grad", layer.bias.shape)
            output, loss, acc = model(x, y)
            self.backprop(model, x, y, output, i)

        return loss, acc

    def backprop(self, model, x, y, output, i):
        # For computing gradients and updating weights
        grad_w, grad_b = self.compute_grads(model, x, y, output)
        self.clip_all_gradients(model, grad_w, grad_b)
        self.update_weights(model, grad_w, grad_b, i)

    def update_weights_SGD(self, model, grad_w, grad_b):
        # Standard update rule for SGD
        for layer in model.layers:
            layer.weights = layer.weights - self.lr * grad_w[layer.n]
            layer.bias = layer.bias - self.lr * grad_b[layer.n].squeeze()

    def update_weights_momentum(self, model, grad_w, grad_b):
        # Initiate history vectors for Momentum
        if self.gradw_his == {} and self.gradb_his == {}:
            for layer in model.layers:
                self.gradw_his[layer.n] = np.zeros_like(layer.weights)
                self.gradb_his[layer.n] = np.zeros_like(layer.bias).reshape(-1, 1)

        # Computing current history vector for Momentum
        for layer in model.layers:
            self.gradw_his[layer.n] = (
                self.beta * self.gradw_his[layer.n] + grad_w[layer.n]
            )
            self.gradb_his[layer.n] = (
                self.beta * self.gradb_his[layer.n] + grad_b[layer.n]
            )

            # Update weights and biases using Momentum
            layer.weights = layer.weights - self.lr * self.gradw_his[layer.n]
            layer.bias = layer.bias - self.lr * self.gradb_his[layer.n].squeeze()

    def update_weights_rmsprop(self, model, grad_w, grad_b, i):
        # Initiating history vectors
        if self.gradw_his == {} and self.gradb_his == {}:
            for layer in model.layers:
                self.gradw_his[layer.n] = np.zeros_like(layer.weights)
                self.gradb_his[layer.n] = np.zeros_like(layer.bias).reshape(-1, 1)

        # Computing current history vector for RMSprop
        for layer in model.layers:
            self.gradw_his[layer.n] = (
                self.beta * self.gradw_his[layer.n]
                + (1 - self.beta) * grad_w[layer.n] ** 2
            )
            self.gradb_his[layer.n].append(
                self.beta * self.gradb_his[layer.n]
                + (1 - self.beta) * grad_b[layer.n] ** 2
            )

            # Update weights and biases using history vector
            layer.weights = (
                layer.weights
                - (self.lr / np.sqrt(self.gradw_his[layer.n] + self.eps))
                * grad_w[layer.n]
            )
            layer.bias = (
                layer.bias
                - (self.lr / np.sqrt(self.gradb_his[layer.n].squeeze() + self.eps))
                * grad_b[layer.n].squeeze()
            )

    def update_weights_adam(self, model, grad_w, grad_b, i):
        # Initiating history vectors
        if self.gradw_m == {} and self.gradb_m == {}:
            for layer in model.layers:
                self.gradw_m[layer.n] = np.zeros_like(layer.weights)
                self.gradb_m[layer.n] = np.zeros_like(layer.bias).reshape(-1, 1)
                self.gradw_his[layer.n] = np.zeros_like(layer.weights)
                self.gradb_his[layer.n] = np.zeros_like(layer.bias).reshape(-1, 1)

        # Computing current history and momentum for Adam
        for layer in model.layers:
            self.gradw_m[layer.n] = (
                self.beta * self.gradw_m[layer.n] + (1 - self.beta) * grad_w[layer.n]
            )
            self.gradw_his[layer.n] = (
                self.beta_2 * self.gradw_his[layer.n]
                + (1 - self.beta_2) * grad_w[layer.n] ** 2
            )

            gradw_mhat = self.gradw_m[layer.n] / (1 - np.power(self.beta, i + 1))
            gradw_vhat = self.gradw_his[layer.n] / (1 - np.power(self.beta_2, i + 1))

            self.gradb_m[layer.n] = (
                self.beta * self.gradb_m[layer.n] + (1 - self.beta) * grad_b[layer.n]
            )
            self.gradb_his[layer.n] = (
                self.beta_2 * self.gradb_his[layer.n]
                + (1 - self.beta_2) * grad_b[layer.n] ** 2
            )

            gradb_mhat = self.gradb_m[layer.n] / ((1 - np.power(self.beta, i + 1)))
            gradb_vhat = self.gradb_his[layer.n] / (1 - np.power(self.beta_2, i + 1))

            # Update weights and biases using new history and momentum
            layer.weights = layer.weights - (self.lr * gradw_mhat) / (
                np.sqrt(gradw_vhat) + self.eps
            )
            layer.bias = layer.bias - (self.lr * gradb_mhat.squeeze()) / (
                np.sqrt(gradb_vhat.squeeze()) + self.eps
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
                self.beta * self.gradw_m[layer.n] + (1 - self.beta) * grad_w[layer.n]
            )
            self.gradw_his[layer.n] = (
                self.beta_2 * self.gradw_his[layer.n]
                + (1 - self.beta_2) * grad_w[layer.n] ** 2
            )

            gradw_mhat = self.gradw_m[layer.n] / (1 - np.power(self.beta, i + 1))
            gradw_vhat = self.gradw_his[layer.n] / (1 - np.power(self.beta_2, i + 1))

            self.gradb_m[layer.n] = (
                self.beta * self.gradb_m[layer.n] + (1 - self.beta) * grad_b[layer.n]
            )
            self.gradb_his[layer.n] = (
                self.beta_2 * self.gradb_his[layer.n]
                + (1 - self.beta_2) * grad_b[layer.n] ** 2
            )

            gradb_mhat = self.gradb_m[layer.n] / ((1 - np.power(self.beta, i + 1)))
            gradb_vhat = self.gradb_his[layer.n] / (1 - np.power(self.beta_2, i + 1))

            layer.weights = layer.weights - (
                self.lr
                * (
                    (self.beta * gradw_mhat)
                    + ((1 - self.beta) * grad_w[layer.n])
                    / (1 - np.power(self.beta, i + 1))
                )
            ) / (np.sqrt(gradw_vhat) + self.eps)
            layer.bias = layer.bias - (
                self.lr
                * (
                    (self.beta * gradb_mhat.squeeze())
                    + ((1 - self.beta) * grad_b[layer.n]).squeeze()
                    / (1 - np.power(self.beta, i + 1))
                )
            ) / (np.sqrt(gradb_vhat.squeeze()) + self.eps)

    def update_weights(self, model, grad_w, grad_b, i):
        if self.optimizer == "sgd":
            self.update_weights_SGD(model, grad_w, grad_b)

        elif self.optimizer == "momentum":
            self.update_weights_momentum(model, grad_w, grad_b)

        elif self.optimizer == "rmsprop":
            self.update_weights_rmsprop(model, grad_w, grad_b)

        elif self.optimizer == "adam":
            self.update_weights_adam(model, grad_w, grad_b, i)

        elif self.optimizer == "nadam":
            self.update_weights_nadam(model, grad_w, grad_b, i)

        if self.optimizer == "sgd":  # standard update rule for SGD
            for layer in model.layers:
                layer.weights = layer.weights - self.lr * grad_w[layer.n]
                layer.bias = layer.bias - self.lr * grad_b[layer.n].squeeze()

        elif self.optimizer in [
            "momentum",
            "nag",
        ]:  # modified update rule for Momentum and NAG
            if self.optimizer == "nag":
                for layer in model.layers:
                    layer.weights = copy.deepcopy(layer.temp_weights)
                    layer.bias = copy.deepcopy(layer.temp_bias)

            if self.gradw_his == {} and self.gradb_his == {}:
                # print("empty")
                for layer in model.layers:
                    self.gradw_his[layer.n] = [np.zeros_like(layer.weights)]
                    self.gradb_his[layer.n] = [np.zeros_like(layer.bias).reshape(-1, 1)]

            for layer in model.layers:
                self.gradw_his[layer.n].append(
                    self.beta * self.gradw_his[layer.n][-1] + grad_w[layer.n]
                )
                # print("gradbhis", self.gradb_his[layer.n][-1].shape, "gradb", grad_b[layer.n].shape)
                self.gradb_his[layer.n].append(
                    self.beta * self.gradb_his[layer.n][-1] + grad_b[layer.n]
                )
                # print(layer.n, "before update", "bias", layer.bias.shape, "bias_his", self.gradb_his[layer.n][-1].shape)

                layer.weights = layer.weights - self.lr * self.gradw_his[layer.n][-1]
                layer.bias = (
                    layer.bias - self.lr * self.gradb_his[layer.n][-1].squeeze()
                )
                # print("after b update", layer.bias.shape)

        elif self.optimizer == "rmsprop":
            if self.gradw_his == {} and self.gradb_his == {}:
                # print("empty")
                for layer in model.layers:
                    self.gradw_his[layer.n] = [np.zeros_like(layer.weights)]
                    self.gradb_his[layer.n] = [np.zeros_like(layer.bias).reshape(-1, 1)]

            for layer in model.layers:
                self.gradw_his[layer.n].append(
                    self.beta * self.gradw_his[layer.n][-1]
                    + (1 - self.beta) * grad_w[layer.n] ** 2
                )
                # print("gradbhis", self.gradb_his[layer.n][-1].shape, "gradb", grad_b[layer.n].shape)
                self.gradb_his[layer.n].append(
                    self.beta * self.gradb_his[layer.n][-1]
                    + (1 - self.beta) * grad_b[layer.n] ** 2
                )
                # print(layer.n, "before update", "bias", layer.bias.shape, "bias_his", self.gradb_his[layer.n][-1].shape)

                layer.weights = (
                    layer.weights
                    - (self.lr / np.sqrt(self.gradw_his[layer.n][-1] + self.eps))
                    * grad_w[layer.n]
                )
                layer.bias = (
                    layer.bias
                    - (
                        self.lr
                        / np.sqrt(self.gradb_his[layer.n][-1].squeeze() + self.eps)
                    )
                    * grad_b[layer.n].squeeze()
                )
                # print("after b update", layer.bias.shape)

        elif self.optimizer == "adam":
            if self.gradw_m == {} and self.gradb_m == {}:
                # print("empty")
                for layer in model.layers:
                    self.gradw_m[layer.n] = [np.zeros_like(layer.weights)]
                    self.gradb_m[layer.n] = [np.zeros_like(layer.bias).reshape(-1, 1)]
                    self.gradw_his[layer.n] = [np.zeros_like(layer.weights)]
                    self.gradb_his[layer.n] = [np.zeros_like(layer.bias).reshape(-1, 1)]

            for layer in model.layers:
                self.gradw_m[layer.n].append(
                    self.beta * self.gradw_m[layer.n][-1]
                    + (1 - self.beta) * grad_w[layer.n]
                )
                self.gradw_his[layer.n].append(
                    self.beta_2 * self.gradw_his[layer.n][-1]
                    + (1 - self.beta_2) * grad_w[layer.n] ** 2
                )
                gradw_mhat = self.gradw_m[layer.n][-1] / (
                    1 - np.power(self.beta, i + 1)
                )
                gradw_vhat = self.gradw_his[layer.n][-1] / (
                    1 - np.power(self.beta_2, i + 1)
                )

                self.gradb_m[layer.n].append(
                    self.beta * self.gradb_m[layer.n][-1]
                    + (1 - self.beta) * grad_b[layer.n]
                )
                self.gradb_his[layer.n].append(
                    self.beta_2 * self.gradb_his[layer.n][-1]
                    + (1 - self.beta_2) * grad_b[layer.n] ** 2
                )

                gradb_mhat = self.gradb_m[layer.n][-1] / (
                    (1 - np.power(self.beta, i + 1))
                )
                gradb_vhat = self.gradb_his[layer.n][-1] / (
                    1 - np.power(self.beta_2, i + 1)
                )
                # print("bias adam", gradb_mhat.shape, gradb_vhat.shape)

                layer.weights = layer.weights - (self.lr * gradw_mhat) / (
                    np.sqrt(gradw_vhat) + self.eps
                )
                layer.bias = layer.bias - (self.lr * gradb_mhat.squeeze()) / (
                    np.sqrt(gradb_vhat.squeeze()) + self.eps
                )

        elif self.optimizer == "nadam":
            if self.gradw_m == {} and self.gradb_m == {}:
                # print("empty")
                for layer in model.layers:
                    self.gradw_m[layer.n] = [np.zeros_like(layer.weights)]
                    self.gradb_m[layer.n] = [np.zeros_like(layer.bias).reshape(-1, 1)]
                    self.gradw_his[layer.n] = [np.zeros_like(layer.weights)]
                    self.gradb_his[layer.n] = [np.zeros_like(layer.bias).reshape(-1, 1)]

            for layer in model.layers:
                self.gradw_m[layer.n].append(
                    self.beta * self.gradw_m[layer.n][-1]
                    + (1 - self.beta) * grad_w[layer.n]
                )
                self.gradw_his[layer.n].append(
                    self.beta_2 * self.gradw_his[layer.n][-1]
                    + (1 - self.beta_2) * grad_w[layer.n] ** 2
                )
                gradw_mhat = self.gradw_m[layer.n][-1] / (
                    1 - np.power(self.beta, i + 1)
                )
                gradw_vhat = self.gradw_his[layer.n][-1] / (
                    1 - np.power(self.beta_2, i + 1)
                )

                self.gradb_m[layer.n].append(
                    self.beta * self.gradb_m[layer.n][-1]
                    + (1 - self.beta) * grad_b[layer.n]
                )
                self.gradb_his[layer.n].append(
                    self.beta_2 * self.gradb_his[layer.n][-1]
                    + (1 - self.beta_2) * grad_b[layer.n] ** 2
                )

                gradb_mhat = self.gradb_m[layer.n][-1] / (
                    (1 - np.power(self.beta, i + 1))
                )
                gradb_vhat = self.gradb_his[layer.n][-1] / (
                    1 - np.power(self.beta_2, i + 1)
                )

                layer.weights = layer.weights - (
                    self.lr
                    * (
                        (self.beta * gradw_mhat)
                        + ((1 - self.beta) * grad_w[layer.n]) / (1 - self.beta)
                    )
                ) / (np.sqrt(gradw_vhat) + self.eps)
                layer.bias = layer.bias - (
                    self.lr
                    * (
                        (self.beta * gradb_mhat.squeeze())
                        + ((1 - self.beta) * grad_b[layer.n]).squeeze()
                    )
                ) / (np.sqrt(gradb_vhat.squeeze()) + self.eps)
                # print("bias nadam", layer.bias.shape)
                # print("term 1", (self.beta*gradb_mhat.squeeze()).shape)
                # print("term 2", ((1-self.beta)* grad_b[layer.n]).shape)
                # print("term 3", (np.sqrt(gradb_vhat.squeeze()) + self.eps).shape)
                # rint("term 4", (self.lr * ((self.beta*gradb_mhat.squeeze()) + ((1-self.beta)* grad_b[layer.n]))/(np.sqrt(gradb_vhat.squeeze()) + self.eps)).shape)

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
                            model.layers[layer.n - 1].act.reshape(
                                batch_size, model.hidden_size, 1
                            ),
                        ),
                        axis=0,
                    )
                    + model.alpha * layer.weights
                )
                grad_b[layer.n] = np.sum(
                    self.grad_bias(grad_a[layer.n]), axis=0
                ) + model.alpha * layer.bias.reshape(-1, 1)

            # Gradients wrt to the hidden layers, preactivation, activation, weights and biases
            else:
                grad_h[layer.n] = self.grad_hidden(
                    model.layers[layer.n + 1].weights, grad_a[layer.n + 1]
                )
                grad_a[layer.n] = self.grad_preact(
                    grad_h[layer.n],
                    model.layers[layer.n].preact.reshape(
                        batch_size, model.hidden_size, 1
                    ),
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
                                model.layers[layer.n - 1].act.reshape(
                                    batch_size, model.hidden_size, 1
                                ),
                            ),
                            axis=0,
                        )
                        + model.alpha * layer.weights
                    )

                grad_b[layer.n] = np.sum(
                    self.grad_bias(grad_a[layer.n]), axis=0
                ) + model.alpha * layer.bias.reshape(-1, 1)

        return grad_w, grad_b

    def grad_yhat(self, y_true, y_pred, loss="cross_entropy"):
        if loss == "cross_entropy":
            y_grad = np.sum(y_true * y_pred, axis=1)
        # elif loss == 'mse':
        #    y_grad = y_pred
        return -(y_true / y_grad[:, np.newaxis])

    def grad_output(self, y_true, y_pred, loss_fn="cross_entropy"):
        if loss_fn.loss_fn == "cross_entropy":
            return -(y_true - y_pred)
        elif loss_fn.loss_fn == "mse":
            grad = []
            for y, y_hat in zip(y_true, y_pred):
                grad.append(
                    y_hat
                    * (
                        (y_hat - y) * (1 - y_hat)
                        - np.tile((y_hat - y).T, (y_hat.shape[0], 1)) @ y_hat
                        + (y_hat - y) * y_hat**2
                    )
                )
            # print(np.array(grad).shape)
            return np.array(grad)

    def grad_hidden(self, W, grad_next_preact):
        # print("W shape", W.shape, "grad_next_preact shape", grad_next_preact.shape)
        # print("W.T shape", W.T.shape)
        # print("grad_hidden shape", (W.T @ grad_next_preact).shape)
        return W.T @ grad_next_preact

    def grad_preact(self, grad_act, act, activation):
        if activation == "sigmoid":  # for sigmoid

            return grad_act * act * (1 - act)
        elif activation == "relu":  # for relu
            return grad_act * (act > 0)
        elif activation == "tanh":  # for tanh
            return grad_act * (1 - act**2)

    def grad_weights(self, grad_preact, input):
        # print("grad preact shape", grad_preact.shape, "input shape", input.shape)
        # print(input.transpose(0, 2, 1).shape)
        # print("grad_weights", (grad_preact @ input.transpose(0, 2, 1)).shape)
        return grad_preact @ input.transpose(0, 2, 1)

    def grad_bias(self, grad_preact):
        return grad_preact

    def clip_gradients(self, gradients):
        return np.clip(gradients, -self.threshold, self.threshold)

    def clip_all_gradients(self, model, grad_w, grad_b):
        for layer in model.layers:
            grad_w[layer.n] = self.clip_gradients(grad_w[layer.n])
            grad_b[layer.n] = self.clip_gradients(grad_b[layer.n])

    def check_for_nans(self, array):
        if np.isnan(array).any():
            raise ValueError("NaN detected in array")

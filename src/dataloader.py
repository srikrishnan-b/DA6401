import numpy as np
import sys
import os
from utils import one_hot_encode, batch_data
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist, mnist


def load_fashion_mnist(batch_size):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Getting the unique labels and one entry from each
    labels = set(y_train)
    sample = {ind: np.where(y_train == ind)[0][0] for ind in labels}

    # Plotting one image from each label
    for i in range(10):
        plt.subplot(4, 3, i + 1)
        plt.imshow(x_train[sample[i]], cmap="gray")
        plt.title(f"label={i}", fontsize=8)

    plt.show()

    # getting validation data
    indices = list(np.arange(len(x_train)))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    x_train, x_valid = x_train[:54000], x_train[54000:]
    y_train, y_valid = y_train[:54000], y_train[54000:]

    # reshaping and normalizing the data
    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_train = x_train / 255.0

    x_valid = x_valid.reshape(x_valid.shape[0], 28 * 28)
    x_valid = x_valid / 255.0

    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_test = x_test / 255.0

    # Converting the labels to one-hot encoding
    y_train = one_hot_encode(y_train)
    y_valid = one_hot_encode(y_valid)
    y_test = one_hot_encode(y_test)

    train = batch_data(x_train, y_train, batch_size)
    valid = batch_data(x_valid, y_valid, batch_size)

    return train, valid, (x_test, y_test)


def load_mnist(batch_size):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Getting the unique labels and one entry from each
    labels = set(y_train)
    sample = {ind: np.where(y_train == ind)[0][0] for ind in labels}

    # Plotting one image from each label
    for i in range(10):
        plt.subplot(4, 3, i + 1)
        plt.imshow(x_train[sample[i]], cmap="gray")
        plt.title(f"label={i}", fontsize=8)

    plt.show()

    # getting validation data
    indices = list(np.arange(len(x_train)))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    x_train, x_valid = x_train[:54000], x_train[54000:]
    y_train, y_valid = y_train[:54000], y_train[54000:]

    # reshaping and normalizing the data
    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_train = x_train / 255.0

    x_valid = x_valid.reshape(x_valid.shape[0], 28 * 28)
    x_valid = x_valid / 255.0

    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_test = x_test / 255.0

    # Converting the labels to one-hot encoding
    y_train = one_hot_encode(y_train)
    y_valid = one_hot_encode(y_valid)
    y_test = one_hot_encode(y_test)

    train = batch_data(x_train, y_train, batch_size)
    valid = batch_data(x_valid, y_valid, batch_size)

    return train, valid(x_test, y_test)

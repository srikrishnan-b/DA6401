import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
import math
import wandb
import argparse
from utils import Loss
from dataloader import load_fashion_mnist, load_mnist
from model import NeuralNet
from optimizer import GradientDescent
from utils import confusion_matrix
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for MLP on Fashion MNIST and MNIST"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="myprojectname", help="Project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="myname",
        help="Entity name to track experiments",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fashion_mnist",
        help="Dataset to train on",
        choices=["fashion_mnist", "mnist"],
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="cross_entropy",
        help="Loss function to use",
        choices=["cross_entropy", "mse"],
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="nadam",
        help="Optimizer to use",
        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0003, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for Momnetum and NAG optimizer",
    )
    parser.add_argument(
        "--beta", type=float, default=0.9, help="Beta for RMSProp optimizer"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="Beta1 for Adam and Nadam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="Beta2 for Adam and Nadam optimizer"
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-8, help="Epsilon for optimizers"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay for optimizers"
    )
    parser.add_argument(
        "--weight_init",
        type=str,
        default="xavier",
        help="Weight initialization for model",
        choices=["random", "xavier"],
    )
    parser.add_argument(
        "--num_layers", type=int, default=5, help="Number of hidden layers in model"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="Hidden size of each layer"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="ReLU",
        help="Activation function to use",
        choices=["identity", "sigmoid", "ReLU", "tanh"],
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    wandb.login()
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_entity,
        config={
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "loss": args.loss,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "momentum": args.momentum,
            "beta": args.beta,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "epsilon": args.epsilon,
            "weight_decay": args.weight_decay,
            "weight_init": args.weight_init,
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "activation": args.activation,
        },
    )

    if args.dataset == "fashion_mnist":
        train, valid, test = load_fashion_mnist(args.batch_size)
    elif args.dataset == "mnist":
        train, valid, test = load_mnist(args.batch_size)
    else:
        raise ValueError("Dataset not supported")

    threshold = 1  # Clipping threshold for gradient clipping
    loss = Loss(loss_fn=args.loss)
    optimizer = GradientDescent(
        optimizer=args.optimizer,
        lr=args.learning_rate,
        clipping_threshold=threshold,
        momentum=args.momentum,
        beta=args.beta,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
    )
    model = NeuralNet(
        input_size=784,
        hidden_size=args.hidden_size,
        output_size=10,
        n_hidden=args.num_layers,
        loss=loss,
        activation=[args.activation, "softmax"],
        weight_init=args.weight_init,
        weight_decay=args.weight_decay,
    )
    i = 0

    for epoch in range(args.epochs):

        batch_train_loss = []
        batch_valid_loss = []
        batch_train_accuracy = []
        batch_valid_accuracy = []

        for x, y in train:
            loss, accuracy = optimizer.optimize(model, x, y, i)
            batch_train_loss.append(loss)
            batch_train_accuracy.append(accuracy)
            wandb.log({"train/batch_loss": loss})

            i += 1

        for x_val, y_val in valid:
            valid_output, valid_loss, valid_accuracy = model(x_val, y_val)
            batch_valid_loss.append(valid_loss)
            batch_valid_accuracy.append(valid_accuracy)

        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": np.mean(batch_train_loss),
                "train/accuracy": np.mean(batch_train_accuracy),
                "val/loss": np.mean(batch_valid_loss),
                "val/accuracy": np.mean(batch_valid_accuracy),
            }
        )
        print(
            f"Epoch: {epoch}, Train Loss: {np.mean(batch_train_loss)}, Valid Loss: {np.mean(batch_valid_loss)}, Valid Accuracy: {np.mean(batch_valid_accuracy)}"
        )

    # Testing model
    _, _, test_acc = model(test[0], test[1])
    wandb.summary["test_accuracy"] = test_acc
    print(f"Test Accuracy: {test_acc}")
    _, _, test_acc = model(test[0], test[1])
    print(f"Test Accuracy: {test_acc}")
    predictions = np.argmax(model.predict(test[0]), axis=1)
    true = np.argmax(test[1], axis=1)
    wandb.summary["test_accuracy"] = test_acc

    # Confusion matrix
    conf = confusion_matrix(true, predictions)
    plt.figure(figsize=(10, 8))
    plt.imshow(conf, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Test set')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(true)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('Truth')
    plt.xlabel('Prediction')
    wandb.log({"Confusion matrix": wandb.Image(plt)})



    wandb.finish()


if __name__ == "__main__":
    main()

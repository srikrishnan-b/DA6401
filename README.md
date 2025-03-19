# DA6401 Assignment 1

Author: Srikrishnan B

This repository contains codes written for Assignment 1 of DA6401. 

## Description

The repository contains codes to implement a feedforward neural network along with implementation backpropogation framework supporting different optimizers. The training and evaluation is done on Fashion-MNIST dataset with provision to use this for MNIST dataset as well. Specifically, the trained model will predict the label of the given 28 * 28 pixel image. 

## Getting started

The codes were written and tested on Python 3. The following packages are required:

- numpy
- pandas
- matplotlib
- keras (for getting the dataset)
- tensorflow (for getting the dataset)
- wandb

Clone this repository; create a conda environment and install the required packages using requirements.txt or environment.yml file located in the root folder. 


## Usage

The training and evaluation of the feedforward neural network can be done either using the python scripts located in the root folder or jupyter notebooks located in the notebooks directory. `train.py` (or `train.ipynb`) is for training a network with a given set of hyperparameters and `hyperparameter_search.py`(or `hyperparameter_search.ipynb`) is for performing a hyperparameter search using the sweep functionality provided by [wandb](https://docs.wandb.ai/guides/sweeps/). `config.json` (used in the notebbook) can be edited to a specific hyperparameter configuration while `sweep_config.json` can be used to set the ranges for hyperparameters to be tuned as well as the sweep type. In the given version of `config.json`, the best hyperparameter values from the experiments is given. 
A sample for how the scripts can be run is given below. All the arguments have default parameters. If you wish to change the parameters, pass them as arguments. 
```python

python train.py --wandbproject "project" --wandb_entity "entity" 

```

```python

python hyperparameter_search.py --wandbproject "project" --count 10

```

## Code Organization
```
├── notebooks/                            # Notebooks for training and sweeps
|   ├── hyperparameter_search.ipynb       # Hyperparameter search
│   ├── train.ipynb                       # Training 
├── src/                                  # Source code directory
│   ├── dataloader.py                     # Loading data
│   ├── model.py                          # Neural network implementation
│   ├── optimizer.py                      # Gradient descent/Backpropogation implementation
│   ├── utils.py                          # Loss functions, activation functions, data preprocessing
├── hyperparameter_search.py              # Hyperparameter search
├── train.py                              # Training
├── sweep_config.json                     # Sweep config file to be used in sweeps
├── config.json                           # Config file to be used in train.ipynb
├── environment.yml                       # Packages
├── requirements.txt                      # Packages
├── README.md                             # Documentation 
```

The `src` directory contains codes used in `train.py` and `hyperparameter_search.py`. 
- `dataloader.py` contains functions to load fashion-MNIST (MNIST) dataset and makes train, validation and test splits; labels are one-hot encoded and batches are created for train and valid splits. 
- `utils.py` contains functions to implement activation functions, loss functions, ond-hot encoding and batch-creation.
- `optimizer.py` implements `GradientDescent` class. `optimizer` function is called using GradientDescent object with the optimizer specified. The following optimizers are supported: SGD, Momentum, NAG, RMSprop, Adam, Nadam. 

The `backprop` function calls the following functions in order: `compute_grads()`, `clip_all_gradients()`, `update_weights()`. Each optimizer has its own rule for updates. `compute_grads()` is common for all optimizers. 

- `model.py` implements two clases: `Layer` and `NeuralNet`. A function called `train_wandb` is defined to be used for wandb sweeps. 

The notebooks follow the same structure as the python scipts. 

---
Link to wandb report: [Link](https://wandb.ai/deeplearn24/DLA1_cross_entropy_sweep/reports/BT23S013-DA6401-Assignment-1--VmlldzoxMTgyNzE1Mg?accessToken=1t2873pn56wpla5n7ro5y6yust2uwo6jnq6m32uykfskm0il20vcr05f339hw7qc)
---
Link to GitHub repo: [Link](https://github.com/srikrishnan-b/DA6401)
---

"""A small autograd engine built for learning."""

from autograd_engine.datasets import load_mnist
from autograd_engine.gradcheck import finite_difference_grad, gradcheck
from autograd_engine.losses import binary_cross_entropy, cross_entropy, mse_loss, softmax
from autograd_engine.metrics import binary_accuracy, multiclass_accuracy
from autograd_engine.nn import Linear, MLP, Module, Parameter, ReLU, Sequential, Sigmoid, Tanh
from autograd_engine.optim import Adam, MomentumSGD, SGD
from autograd_engine.scalar import Value
from autograd_engine.serialization import (
    load_state_dict,
    load_state_dict_from_file,
    save_state_dict,
    state_dict,
)
from autograd_engine.tensor import Tensor
from autograd_engine.training import DataLoader, Dataset, History, TensorDataset, evaluate, train_epoch

__all__ = [
    "Adam",
    "DataLoader",
    "Dataset",
    "History",
    "Linear",
    "MLP",
    "Module",
    "MomentumSGD",
    "Parameter",
    "ReLU",
    "SGD",
    "Sequential",
    "Sigmoid",
    "Tanh",
    "Tensor",
    "TensorDataset",
    "Value",
    "binary_accuracy",
    "binary_cross_entropy",
    "cross_entropy",
    "evaluate",
    "finite_difference_grad",
    "gradcheck",
    "load_state_dict",
    "load_state_dict_from_file",
    "load_mnist",
    "mse_loss",
    "multiclass_accuracy",
    "save_state_dict",
    "softmax",
    "state_dict",
    "train_epoch",
]

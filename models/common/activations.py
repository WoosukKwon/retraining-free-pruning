import torch
import torch.nn as nn


ACT2FN = {
    "relu": nn.functional.relu,
    "gelu": nn.functional.gelu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}

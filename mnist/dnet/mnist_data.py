import os
from pathlib import Path

import jax.numpy as tensor
import pandas as pd


def tiny_mnist(flatten: bool = True, one_hot_encoding: bool = True,
               data_dir: str = os.path.join("datasets", "tiny_mnist")):
    path = Path(data_dir)
    train_data = tensor.asarray(pd.read_csv(path / "train.csv", header=None).values, dtype=tensor.float32)
    valid_data = tensor.asarray(pd.read_csv(path / "test.csv", header=None).values, dtype=tensor.float32)
    train_images = train_data[:, 1:] / 255.0
    train_labels = train_data[:, 0].reshape(-1, 1)
    valid_images = valid_data[:, 1:] / 255.0
    valid_labels = valid_data[:, 0].reshape(-1, 1)
    if not flatten:
        train_images = train_images.reshape(-1, 28, 28, 1)
        valid_images = valid_images.reshape(-1, 28, 28, 1)
    if one_hot_encoding:
        train_labels = tensor.asarray(pd.get_dummies(train_labels), dtype=tensor.float32)
        valid_labels = tensor.asarray(pd.get_dummies(valid_labels), dtype=tensor.float32)
    return (train_images, train_labels), (valid_images, valid_labels)

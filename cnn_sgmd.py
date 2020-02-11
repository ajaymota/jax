# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A basic MNIST example using JAX with the mini-libraries stax and optimizers.

The mini-library jax.experimental.stax is for neural network building, and
the mini-library jax.experimental.optimizers is for first-order stochastic
optimization.
"""

from dnet import mnist_data
import time
import itertools

import pandas as pd

import numpy.random as npr
import numpy as onp

import jax.numpy as np
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Sigmoid, LogSoftmax, Conv, MaxPool, Flatten

eigenvalues0 = []
eigenvalues1 = []
eigenvalues2 = []
eigenvalues3 = []
eigenvalues4 = []
eigenvalues5 = []
eigenvalues6 = []
eigenvalues7 = []
eigenvalues8 = []
eigenvalues9 = []

train_loss_arr = []
test_loss_arr = []
test_class_loss_arr = []


def get_eigenvalues(matrix):
    return np.linalg.eigvalsh(matrix)


def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return -np.mean(np.sum(preds * targets, axis=1))


def accuracy(params, batch):
    inputs, targets = batch
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)


init_random_params, predict = stax.serial(
    Conv(10, (5, 5), (1, 1)), Sigmoid,
    MaxPool((4, 4)), Flatten,
    Dense(10), LogSoftmax)

if __name__ == "__main__":
    rng = random.PRNGKey(0)
    
    step_size = 0.001
    num_epochs = 10
    batch_size = 128
    momentum_mass = 0.9

    # input shape for CNN
    input_shape = (-1, 28, 28, 1)
    
    # training/test split
    (train_images, train_labels), (test_images, test_labels) = mnist_data.tiny_mnist(flatten=False)
    num_train = train_images.shape[0]
    
    print(train_images.shape)
    
    # converting to batches
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    
    
    # randomize batches and store them
    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]
    
    
    batches = data_stream()
    
    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)
    
    
    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)
    
    
    _, init_params = init_random_params(rng, input_shape)
    opt_state = opt_init(init_params)
    itercount = itertools.count()
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batches))
        epoch_time = time.time() - start_time
        
        params = get_params(opt_state)
        train_acc = accuracy(params, (train_images, train_labels))
        test_acc = accuracy(params, (test_images, test_labels))
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))
        
        # print(len(params))
        # print(params[0][0].shape)
        # print(params[0][0])
        # x = params[0][0].reshape(5, 5, 10)
        # # print(np.transpose(x))
        # print(np.transpose(x).swapaxes(1, 2))
        # break
        
        train_loss = loss(params, (train_images, train_labels))
        test_loss = loss(params, (test_images, test_labels))
        print("Training set loss {}".format(train_loss))
        print("Test set loss {}".format(test_loss))

        train_loss_arr.append(train_loss)
        test_loss_arr.append(test_loss)
        test_class_loss_arr.append(1-test_acc)
        print()
        
        kernels = np.transpose(params[0][0].reshape(5, 5, 10)).swapaxes(1, 2)

        # params[0][0].shape = (5, 5, 1, 10)    -> Input edge weights
        # params[4][1].shape = (10, )           -> Output layer
        eigv0 = onp.asarray(get_eigenvalues(kernels[0]))
        eigv1 = onp.asarray(get_eigenvalues(kernels[1]))
        eigv2 = onp.asarray(get_eigenvalues(kernels[2]))
        eigv3 = onp.asarray(get_eigenvalues(kernels[3]))
        eigv4 = onp.asarray(get_eigenvalues(kernels[4]))
        eigv5 = onp.asarray(get_eigenvalues(kernels[5]))
        eigv6 = onp.asarray(get_eigenvalues(kernels[6]))
        eigv7 = onp.asarray(get_eigenvalues(kernels[7]))
        eigv8 = onp.asarray(get_eigenvalues(kernels[8]))
        eigv9 = onp.asarray(get_eigenvalues(kernels[9]))

        eigenvalues0.append(eigv0)
        eigenvalues1.append(eigv1)
        eigenvalues2.append(eigv2)
        eigenvalues3.append(eigv3)
        eigenvalues4.append(eigv4)
        eigenvalues5.append(eigv5)
        eigenvalues6.append(eigv6)
        eigenvalues7.append(eigv7)
        eigenvalues8.append(eigv8)
        eigenvalues9.append(eigv9)

    df0 = pd.DataFrame(data=eigenvalues0)
    df1 = pd.DataFrame(data=eigenvalues1)
    df2 = pd.DataFrame(data=eigenvalues2)
    df3 = pd.DataFrame(data=eigenvalues3)
    df4 = pd.DataFrame(data=eigenvalues4)
    df5 = pd.DataFrame(data=eigenvalues5)
    df6 = pd.DataFrame(data=eigenvalues6)
    df7 = pd.DataFrame(data=eigenvalues7)
    df8 = pd.DataFrame(data=eigenvalues8)
    df9 = pd.DataFrame(data=eigenvalues9)
    
    df10 = pd.DataFrame(data=train_loss_arr)
    df11 = pd.DataFrame(data=test_loss_arr)
    df12 = pd.DataFrame(data=test_class_loss_arr)

    df0.to_csv("results/cnn_sgmd_K0.csv", index_label=False)
    df1.to_csv("results/cnn_sgmd_K1.csv", index_label=False)
    df2.to_csv("results/cnn_sgmd_K2.csv", index_label=False)
    df3.to_csv("results/cnn_sgmd_K3.csv", index_label=False)
    df4.to_csv("results/cnn_sgmd_K4.csv", index_label=False)
    df5.to_csv("results/cnn_sgmd_K5.csv", index_label=False)
    df6.to_csv("results/cnn_sgmd_K6.csv", index_label=False)
    df7.to_csv("results/cnn_sgmd_K7.csv", index_label=False)
    df8.to_csv("results/cnn_sgmd_K8.csv", index_label=False)
    df9.to_csv("results/cnn_sgmd_K9.csv", index_label=False)
    df10.to_csv("results/cnn_sgmd_train_loss.csv", index_label=False)
    df11.to_csv("results/cnn_sgmd_test_loss.csv", index_label=False)
    df12.to_csv("results/cnn_sgmd_test_class_loss.csv", index_label=False)


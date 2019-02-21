import tensorflow as tf
import numpy as np
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Load the data
def load_data():
    with np.load("notMNIST.npz") as data:
        data, target = data["images"], data["labels"]
        np.random.seed(521)
        rand_indx = np.arange(len(data))
        np.random.shuffle(rand_indx)
        data = data[rand_indx] / 255.0
        target = target[rand_indx]
        train_data, train_target = data[:10000].reshape((-1, 28 * 28)), target[:10000]
        valid_data, valid_target = data[10000:16000].reshape((-1, 28 * 28)), target[10000:16000]
        test_data, test_target = data[16000:].reshape((-1, 28 * 28)), target[16000:]
    return train_data, valid_data, test_data, train_target, valid_target, test_target


# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convert_one_hot(train_target, valid_target, test_target):
    new_train = np.zeros((train_target.shape[0], 10))
    new_valid = np.zeros((valid_target.shape[0], 10))
    new_test = np.zeros((test_target.shape[0], 10))

    for item in range(0, train_target.shape[0]):
        new_train[item][train_target[item]] = 1
    for item in range(0, valid_target.shape[0]):
        new_valid[item][valid_target[item]] = 1
    for item in range(0, test_target.shape[0]):
        new_test[item][test_target[item]] = 1
    return new_train, new_valid, new_test


def shuffle(train_data, train_target):
    np.random.seed(421)
    rand_indx = np.arange(len(train_data))
    target = train_target
    np.random.shuffle(rand_indx)
    data, target = train_data[rand_indx], target[rand_indx]
    return data, target


def relu(x):
    return x.clip(min=0)


def softmax(x):
    x = np.exp(x)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def compute_layer(x, w, b):
    return np.dot(x, w) + b


def average_ce(target, prediction):
    ce = (target * np.log(prediction)).sum(axis=1)
    return -ce.mean()


def grad_ce(target, prediction):
    pass

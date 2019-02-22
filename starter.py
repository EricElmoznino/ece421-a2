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


def linear(x, w, b):
    return np.dot(x, w) + b


def average_ce(target, prediction):
    ce = (target * np.log(prediction)).sum(axis=1)
    return -ce.mean()


def grad_ce(target, prediction):
    return 1 / len(target) * (prediction - target)


def grad_w(delta, x):
    return np.dot(delta.T, x)


def grad_b(delta):
    return delta.sum(axis=0, keepdims=True)


def grad_x(delta, w):
    return np.dot(delta, w.T)


def grad_relu(delta, s):
    g = delta.copy()
    g[s < 0] = 0
    return g


class Network:

    def __init__(self, hidden_size):
        self.w_h = init_weights((28 * 28, hidden_size))
        self.b_h = init_biases(hidden_size)
        self.w_o = init_weights((hidden_size, 10))
        self.b_o = init_biases(10)

        self.w_h_grad = None
        self.b_h_grad = None
        self.w_o_grad = None
        self.b_o_grad = None

        self.input = None
        self.s_h = None
        self.x_h = None

    def forward(self, input):
        s_h = linear(input, self.w_h, self.b_h)
        x_h = relu(s_h)
        s_o = linear(x_h, self.w_o, self.b_o)
        s_x = softmax(s_o)

        self.input = input
        self.s_h = s_h
        self.x_h = x_h

        return s_x

    def backward(self, target, prediction):
        s_o_grad = grad_ce(target, prediction)
        self.w_o_grad = grad_w(s_o_grad, self.x_h)
        self.b_o_grad = grad_b(s_o_grad)
        x_h_grad = grad_x(s_o_grad, self.w_o)
        s_h_grad = grad_relu(x_h_grad, self.s_h)
        self.w_h_grad = grad_w(s_h_grad, self.input)
        self.b_h_grad = grad_b(s_h_grad)


class Optimizer:

    def __init__(self, lr, momentum, network):
        self.lr = lr
        self.momentum = momentum
        self.network = network

        self.w_h_v = 0
        self.b_h_v = 0
        self.w_o_v = 0
        self.b_o_v = 0

    def step(self):
        self.w_h_v = self.momentum * self.w_h_v + self.lr * self.network.w_h_grad
        self.network.w_h -= self.w_h_v
        self.b_h_v = self.momentum * self.b_h_v + self.lr * self.network.b_h_grad
        self.network.b_h -= self.b_h_v
        self.w_o_v = self.momentum * self.w_o_v + self.lr * self.network.w_o_grad
        self.network.w_o -= self.w_o_v
        self.b_o_v = self.momentum * self.b_o_v + self.lr * self.network.b_o_grad
        self.network.b_o -= self.b_o_v


def init_weights(shape):
    return np.random.normal(loc=0, scale=np.sqrt(2 / (shape[0] + shape[1])), size=shape)


def init_biases(size):
    return np.random.normal(loc=0, scale=np.sqrt(2 / size), size=(1, size))

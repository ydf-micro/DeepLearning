import sklearn
import numpy as np
from testCases import *
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return n_x, n_h, n_y

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (w1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (w2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2
    }

    return parameters

def forword_propagation(X, parameters):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    Z1 = np.dot(w1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(w2, A1) + b2
    A2 = sigmoid(Z2)

    assert A2.shape == (1, X.shape[1])
    cache = {
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2
    }

    return A2, cache

def compute_cost(A2, Y, parameters):
    m = Y.shape[1]

    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = -np.sum(logprobs) / m
    cost = float(np.squeeze(cost))

    assert isinstance(cost, float)

    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    w1 = parameters['w1']
    w2 = parameters['w2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dw2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(w2.T, dZ2), 1 - np.power(A1, 2))
    dw1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        'dw1': dw1,
        'db1': db1,
        'dw2': dw2,
        'db2': db2
    }

    return grads

def update_parameters(parameters, grads, alpha=1.2):
    w1, w2 = parameters['w1'], parameters['w2']
    b1, b2 = parameters['b1'], parameters['b2']

    dw1, dw2  = grads['dw1'], grads['dw2']
    db1, db2 = grads['db1'], grads['db2']

    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2

    parameters = {
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2
    }

    return parameters

def nn_model(X, Y, n_h, num_iterations, print_cost=False):
    np.random.seed(3)
    n_x, _, n_y = layer_sizes(X, Y)

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(num_iterations):
        A2, cache = forword_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, alpha=0.5)

        if print_cost:
            if i % 100 == 0:
                print('第', i, '次循环，成本为: ' + str(cost))

    return parameters

def predict(parameters, X):
    A2, cache = forword_propagation(X, parameters)
    predictions = np.round(A2)

    return predictions

if __name__ == '__main__':
    X, Y = load_planar_dataset()

    # plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
    # print(X.shape, Y.shape)
    # clf = sklearn.linear_model.LogisticRegressionCV()
    # clf.fit(X.T, Y.T)
    # plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    # plt.title('Logistic Regression')
    # LR_predictions = clf.predict(X.T)
    # print('逻辑回归的准确性: %d '% float((np.dot(Y, LR_predictions) +
    #                                   np.dot(1-Y, 1-LR_predictions)) / float(Y.size) * 100) + '% ' + '(正确标记的数据点占的百分比)')
    # plt.show()

    # print('==================================测试layer_sizes==================================')
    # X_asses, Y_asses = layer_sizes_test_case()
    # n_x, n_h, n_y = layer_sizes(X_asses, Y_asses)
    #
    # print(n_x, n_h, n_y)

    # print('==================================测试initialize_parameters==================================')
    # n_x, n_h, n_y = initialize_parameters_test_case()
    # parameters = initialize_parameters(n_x, n_h, n_y)
    # print(parameters['w1'])
    # print(parameters['b1'])
    # print(parameters['w2'])
    # print(parameters['b2'])

    # print('==================================测试forward_propagation==================================')
    # X_assess, parameters = forward_propagation_test_case()
    # A2, cache = forword_propagation(X_assess, parameters)
    # print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))

    # A2, Y_assess, parameters = compute_cost_test_case()
    # print(compute_cost(A2, Y_assess, parameters))

    plt.figure(figsize=(32, 16))
    hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
    for i, n_h in enumerate(hidden_layer_sizes):
        plt.subplot(4, 2, i + 1)
        plt.title('Hidden Layer of size %d' % n_h)
        parameters = nn_model(X, Y, n_h, num_iterations=5000)

        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
        print('隐藏层的节点数量: {}, 准确率: {}%'.format(n_h, accuracy))
    plt.show()
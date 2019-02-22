import sklearn
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import init_utils
import reg_utils
import gc_utils

def model(X, Y, alpha=0.3, num_iterations=30000, print_cost=True, lambd=0.0, keep_prob=1):
    grads = {}
    costs = []
    layers_dims = [X.shape[0], 20, 3, 1]

    parameters = reg_utils.initialize_parameters(layers_dims)

    for i in range(num_iterations):
        if keep_prob == 1:
            a3, cache = reg_utils.forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        else:
            print('keep_prob参数错误！程序退出。')

        if lambd == 0:
            cost = reg_utils.compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

        assert lambd == 0 or keep_prob == 1

        if lambd == 0 and keep_prob == 1:
            grads = reg_utils.backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        parameters = reg_utils.update_parameters(parameters, grads, alpha)

        if i % 1000 == 0:
            costs.append(cost)
            if print_cost and i % 10000 == 0:
                print('第', i, '次迭代，成本值为: ', cost)

    return parameters, costs, alpha

def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
    np.random.seed(1)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]


    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    z1 = np.dot(W1, X) + b1
    a1 = reg_utils.relu(z1)
    d1 = np.random.rand(a1.shape[0], a1.shape[1])
    d1 = d1 < keep_prob
    a1 *= d1
    a1 /= keep_prob
    z2 = np.dot(W2, a1) + b2
    a2 = reg_utils.relu(z2)
    d2 = np.random.rand(a2.shape[0], a2.shape[1])
    d2 = d2 < keep_prob
    a2 *= d2
    a2 /= keep_prob
    z3 = np.dot(W3, a2) + b3
    a3 = reg_utils.sigmoid(z3)

    cache = (z1, d1, a1, W1, b1, z2, d2, a2, W2, b2, z3, a3, W3, b3)

    return a3, cache

def compute_cost_with_regularization(A3, Y, parameters, lambd):
    m = Y.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    cross_entropy_cost = reg_utils.compute_cost(A3, Y)
    L2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2 * m)
    cost = cross_entropy_cost + L2_regularization_cost

    return cost

def backward_propagation_with_regularization(X, Y, cache, lambd):
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache

    dz3 = 1. / m * (a3 - Y)
    dW3 = np.dot(dz3, a2.T) + lambd * W3 / m
    db3 = np.sum(dz3, axis=1, keepdims=True)

    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T) + lambd * W2 / m
    db2 = np.sum(dz2, axis=1, keepdims=True)

    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T) + lambd * W1 / m
    db1 = np.sum(dz1, axis=1, keepdims=True)

    gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                 "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}

    return gradients

def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    m = X.shape[1]
    (z1, d1, a1, W1, b1, z2, d2, a2, W2, b2, z3, a3, W3, b3) = cache

    dz3 = 1. / m * (a3 - Y)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims=True)

    da2 = np.dot(W3.T, dz3)
    da2 *= d2 / keep_prob
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)

    da1 = np.dot(W2.T, dz2)
    da1 *= d1 / keep_prob
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                 "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}

    return gradients

if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = reg_utils.load_2D_dataset(False)
    parameters, costs, alpha = model(train_X, train_Y, keep_prob=0.86, alpha=0.3)

    # plt.plot(costs)
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per hundreds)')
    # plt.title('Learning rate = ' + str(alpha))
    # plt.show()
    #
    # print('训练集:')
    # predictions_train = reg_utils.predict(train_X, train_Y, parameters)
    # print('测试集:')
    # predictions_test = reg_utils.predict(test_X, test_Y, parameters)

    axes = plt.gca()
    axes.set_xlim([-0.75, 0.4])
    axes.set_ylim([-0.75, 0.65])
    reg_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)
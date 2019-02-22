import numpy as np
import matplotlib.pyplot as plt
import lr_utils
import testCases
from skimage import transform
import imageio
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward

def initialize_parameters(n_x, n_h, n_y):
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

def initialize_parameters_deep(layers_dims):
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['w' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) / np.sqrt(layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        assert parameters['w' + str(l)].shape == (layers_dims[l], layers_dims[l-1])
        assert parameters['b' + str(l)].shape == (layers_dims[l], 1)

    return parameters

def linear_forward(A, w, b):
    Z = np.dot(w, A) + b
    assert Z.shape == (w.shape[0], A.shape[1])
    cache = (A, w, b)

    return Z, cache

def linear_activation_forward(A_prev, w, b, activation):
    Z, linear_cache = linear_forward(A_prev, w, b)
    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        A, activation_cache = relu(Z)

    assert A.shape == (w.shape[0], A_prev.shape[1])
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['w' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['w' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)

    assert AL.shape == (1, X.shape[1])

    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m
    cost = np.squeeze(cost)
    assert cost.shape == ()

    return cost

def linear_backward(dZ, cache):
    A_prev, w, b = cache
    m = A_prev.shape[1]
    dw = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(w.T, dZ)

    assert dA_prev.shape == A_prev.shape
    assert dw.shape == w.shape
    assert db.shape == b.shape

    return dA_prev, dw, db

def linear_activation_backward(dA, cache, activation='relu'):
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    dA_prev, dw, db = linear_backward(dZ, linear_cache)

    return dA_prev, dw, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = -np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)

    current_cache = caches[-1]
    grads['dA' + str(L-1)], grads['dw' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads['dA' + str(l)], grads['dw' + str(l + 1)], grads['db' + str(l + 1)] = \
            linear_activation_backward(grads['dA' + str(l + 1)], current_cache, 'relu')

    return grads

def update_parameters(parameters, grads, alpha):
    L = len(parameters) // 2
    for l in range(L):
        parameters['w' + str(l + 1)] = parameters['w' + str(l + 1)] - alpha * grads['dw' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - alpha * grads['db' + str(l + 1)]

    return parameters

def two_layer_model(X, Y, layers_dims, alpha=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    grads = {}
    costs = []
    n_x, n_h, n_y = layers_dims

    parameters = initialize_parameters(n_x, n_h, n_y)

    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    for i in range(0, num_iterations):
        A1, cache1 = linear_activation_forward(X, w1, b1, 'relu')
        A2, cache2 = linear_activation_forward(A1, w2, b2, 'sigmoid')

        cost = compute_cost(A2, Y)

        dA2 = -np.divide(Y, A2) + np.divide(1 - Y, 1 - A2)

        dA1, dw2, db2 = linear_activation_backward(dA2,  cache2, 'sigmoid')
        dA0, dw1, db1 = linear_activation_backward(dA1, cache1, 'relu')

        grads['dw1'] = dw1
        grads['db1'] = db1
        grads['dw2'] = dw2
        grads['db2'] = db2
        # print(grads)

        parameters = update_parameters(parameters, grads, alpha)
        w1 = parameters['w1']
        b1 = parameters['b1']
        w2 = parameters['w2']
        b2 = parameters['b2']
        # print(parameters)

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print('第', i, '次迭代，成本值为: ', np.squeeze(cost))

    return parameters, costs, alpha

def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))

    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    print('准确度: ' + str(float(np.sum(p == y)) / m))

    return p

def L_layer_model(X, Y, layers_dims, alpha=0.0075, num_iterations=3000, print_cost=False, isPlot=True):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, alpha)

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print('第', i, '次迭代，成本值: ', np.squeeze(cost))

    return parameters, costs, alpha

def print_mislabelled_images(classes, X, y, p):
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (50.0, 50.0)
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title('Pre: ' + classes[int(p[0, index])].astype('U13') + '\n Class: ' + classes[y[0, index]].astype('U13'))

if __name__ == '__main__':
    # layers_dims = [5, 4, 3]
    # parameters = initialize_parameters_deep(layers_dims)
    # print(parameters['w1'])
    # print(parameters['b1'])
    # print(parameters['w2'])
    # print(parameters['b2'])

    # A, w, b = testCases.linear_forward_test_case()
    # Z, linear_cache = linear_forward(A, w, b)
    # print(Z)
    # print(linear_cache)

    # A_prev, w, b = testCases.linear_activation_forward_test_case()
    #
    # A, linear_activation_cache = linear_activation_forward(A_prev, w, b, activation='sigmoid')
    # print(A)
    #
    # A, linear_activation_cache = linear_activation_forward(A_prev, w, b, activation='relu')
    # print(A)

    # X, parameters = testCases.L_model_forward_test_case()
    # AL, caches = L_model_forward(X, parameters)
    # print(AL)
    # print(len(caches))

    # train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()
    #
    # train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    # test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    #
    # train_x = train_x_flatten / 255
    # train_y = train_set_y
    # test_x = test_x_flatten / 255
    # test_y = test_set_y
    #
    # n_x = train_x.shape[0]
    # n_h = 7
    # n_y = 1
    # layers_dims = (n_x, n_h, n_y)
    #
    # parameters, costs, alpha = two_layer_model(train_x, train_set_y, layers_dims=layers_dims, num_iterations=2500, print_cost=True)
    #
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title('Learning rate = ' + str(alpha))
    # plt.show()
    #
    # predictions_train = predict(train_x, train_y, parameters)
    # predictions_test = predict(test_x, test_y, parameters)


    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()

    train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_x = train_x_flatten / 255
    train_y = train_set_y
    test_x = test_x_flatten / 255
    test_y = test_set_y

    layers_dims = [train_x.shape[0], 20, 7, 5, 1]

    parameters, costs, alpha = L_layer_model(train_x, train_set_y, layers_dims=layers_dims, num_iterations=2500, print_cost=True)

    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title('Learning rate = ' + str(alpha))


    predictions_train = predict(train_x, train_y, parameters)
    predictions_test = predict(test_x, test_y, parameters)

    print_mislabelled_images(classes, test_x, test_y, predictions_test)
    plt.show()
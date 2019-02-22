import sklearn
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import init_utils
import reg_utils
import gc_utils

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def model(X, Y, alpha=0.01, num_iterations=15000, print_cost=True, initialization='he'):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]

    if initialization == 'zeros':
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == 'random':
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == 'he':
        parameters = initialize_parameters_he(layers_dims)
    else:
        print('错误初始化参数！程序退出')
        exit()

    for i in range(num_iterations):
        a3, cache = init_utils.forward_propagation(X, parameters)
        cost = init_utils.compute_loss(a3, Y)
        grads = init_utils.backward_propagation(X, Y, cache)
        parameters = init_utils.update_parameters(parameters, grads, alpha)

        if i % 1000 == 0:
            costs.append(cost)
            if print_cost:
                print('第', i, '次迭代，成本值为: ', cost)

    return parameters, costs, alpha

def initialize_parameters_zeros(layers_dims):
    parameters = {}

    L = len(layers_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        assert parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l-1])
        assert parameters['b' + str(l)].shape == (layers_dims[l], 1)

    return parameters

def initialize_parameters_random(layers_dims):
    parameters = {}

    L = len(layers_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        assert parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l - 1])
        assert parameters['b' + str(l)].shape == (layers_dims[l], 1)

    return parameters

def initialize_parameters_he(layers_dims):
    parameters = {}

    L = len(layers_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.rand(layers_dims[l], layers_dims[l - 1]) / np.sqrt(2 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        assert parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l - 1])
        assert parameters['b' + str(l)].shape == (layers_dims[l], 1)

    return parameters

if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = init_utils.load_dataset(False)
    parameters, costs, alpha = model( train_X, train_Y, initialization='he')

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title('Learning rate = ' + str(alpha))
    plt.show()
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)

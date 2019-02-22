import random
import numpy as np
import matplotlib.pyplot as plt
from lr_utils import load_dataset

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))

    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1-Y) * (np.log(1 - A)))

    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)

    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {'dw': dw, 'db': db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, alpha, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads['dw']
        db = grads['db']

        w = w - alpha * dw
        b = b - alpha * db

        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print('迭代次数: %d, 误差值: %f' % (i, cost))

    params = {'w': w, 'b': b}
    grads = {'dw': dw, 'db': db}

    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    assert(Y_prediction.shape == (1, m))

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, alpha = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, alpha, print_cost)

    w, b = parameters['w'], parameters['b']

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print('训练集的准确性: {}%'.format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print('测试集的准确性: {}%'.format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {
        'costs': costs,
        'Y_prediction_test': Y_prediction_test,
        'Y_prediction_train': Y_prediction_train,
        'w': w,
        'b': b,
        'alpha': alpha,
        'num_iterations': num_iterations
    }

    return d

if __name__ == '__main__':
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    # index = random.randint(0, 209)
    # plt.imshow(train_set_x_orig[index])
    # plt.show()
    # print('y = ' + str(train_set_y[:, index]) +
    #                    ", it`s a '" + classes[np.squeeze(train_set_y[:, index])].astype('U13') +
    #                    "' picture.")

    # m_train = train_set_y.shape[1]
    # m_test = test_set_y.shape[1]
    # num_px = train_set_x_orig.shape[1]
    #
    # print('训练集的数量： m_train: ' + str(m_train))
    # print("测试集的数量 : m_test = " + str(m_test))
    # print("每张图片的宽/高 : num_px = " + str(num_px))
    # print("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
    # print("训练集_图片的维数 : " + str(train_set_x_orig.shape))
    # print("训练集_标签的维数 : " + str(train_set_y.shape))
    # print("测试集_图片的维数: " + str(test_set_x_orig.shape))
    # print("测试集_标签的维数: " + str(test_set_y.shape))

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    # print("\n训练集降维最后的维度： " + str(train_set_x_flatten.shape))
    # print("训练集_标签的维数 : " + str(train_set_y.shape))
    # print("测试集降维之后的维度: " + str(test_set_x_flatten.shape))
    # print("测试集_标签的维数 : " + str(test_set_y.shape))

    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255

    # w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
    # grads, cost = propagate(w, b, X, Y)
    # print(grads['dw'])
    # print(grads['db'])
    # print(cost)

    print('============================测试model============================')
    # d = models(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, alpha=0.005, print_cost=True)
    #
    # costs = np.squeeze(d['costs'])
    # plt.plot(costs)
    # plt.ylabel('cost')
    # plt.xlabel('iterations(per hundreds)')
    # plt.title('learning_rate' + str(d['alpha']))
    # plt.show()

    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    for i in learning_rates:
        print('learning rate is: ' + str(i))
        models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, alpha=i, print_cost=True)
        print('\n------------------------------------------------------------------------------\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]['costs']), label = str(models[str(i)]['alpha']))

    plt.ylabel('cost')
    plt.xlabel('iterations(per hundreds)')
    legend = plt.legend(loc='upper right', ncol=1, shadow=True)
    plt.show()
# *_*coding:utf-8 *_*

import numpy as np
import rnn_utils

def rnn_cell_froward(xt, a_prev, parameters):
    '''

    :param xt:
    :param a_prev:
    :param parameters: Wax, Waa, Wya, ba, by
    :return:
    '''

    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']

    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)

    yt_pred = rnn_utils.softmax(np.dot(Wya, a_next) + by)

    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache

def rnn_forward(x, a0, parameters):
    '''

    :param x:
    :param a0:
    :param parameters:
    :return:
    '''

    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wya'].shape

    a = np.zeros([n_a, m, T_x])
    y_pred = np.zeros([n_y, m, T_x])

    a_next = a0

    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_froward(x[:, :, t], a_next, parameters)
        a[:, :, t] = a_next
        y_pred[:, :, t] = yt_pred
        caches.append(cache)

    caches = (caches, x)

    return a, y_pred, caches

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    '''

    :param xt:
    :param a_prev:
    :param c_prev:
    :param parameters:
    :return:
    '''

    Wf = parameters['Wf']
    bf = parameters['bf']
    Wi = parameters['Wi']
    bi = parameters['bi']
    Wc = parameters['Wc']
    bc = parameters['bc']
    Wo = parameters['Wo']
    bo = parameters['bo']
    Wy = parameters['Wy']
    by = parameters['by']

    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    contact = np.zeros([n_a + n_x, m])
    contact[: n_a, :] = a_prev
    contact[n_a:, :] = xt

    ft = rnn_utils.sigmoid(np.dot(Wf, contact) + bf)

    it = rnn_utils.sigmoid(np.dot(Wi, contact) + bi)

    cct = np.tanh(np.dot(Wc, contact) + bc)

    c_next = ft * c_prev + it * cct

    ot = rnn_utils.sigmoid(np.dot(Wo, contact) + bo)

    a_next = ot * np.tanh(c_next)

    yt_pred = rnn_utils.softmax(np.dot(Wy, a_next) + by)

    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache

def lstm_forward(x, a0, parameters):
    '''

    :param x:
    :param a0:
    :param parameters:
    :return:
    '''

    caches = []

    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wy'].shape

    a = np.zeros([n_a, m, T_x])
    c = np.zeros([n_a, m, T_x])
    y = np.zeros([n_y, m, T_x])

    a_next = a0
    c_next = np.zeros([n_a, m])

    for t in range(T_x):
        a_next, c_next, yt_pred, cache = lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)

        a[:, :, t] = a_next

        y[:, :, t] = yt_pred

        c[:, :, t] = c_next

        caches.append(cache)

    caches = (caches, x)

    return a, y, c, caches

def rnn_cell_backwawrd(da_next, cache):
    '''

    :param da_next:
    :param cache:
    :return:
    '''

    a_next, a_prev, xt, parameters = cache

    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']

    dtanh = (1 - np.square(np.tanh(a_next))) * da_next

    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)

    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)

    dba = np.sum(dtanh, keepdims=True, axis=-1)

    gradients = {'dxt': dxt, 'da_prev': da_prev, 'dWax': dWax, 'dWaa': dWaa, 'dba': dba}

    return gradients

if __name__ == '__main__':
    np.random.seed(1)
    x = np.random.randn(3, 10, 7)
    a0 = np.random.randn(5, 10)
    Wf = np.random.randn(5, 5+3)
    bf = np.random.randn(5, 1)
    Wi = np.random.randn(5, 5+3)
    bi = np.random.randn(5, 1)
    Wc = np.random.randn(5, 5+3)
    bc = np.random.randn(5, 1)
    Wo = np.random.randn(5, 5+3)
    bo = np.random.randn(5, 1)
    Wy = np.random.randn(2, 5)
    by = np.random.randn(2, 1)
    parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

    a, y, c, caches = lstm_forward(x, a0, parameters)

    print("a[4][3][6] = ", a[4][3][6])
    print("a.shape = ", a.shape)
    print("y[1][4][3] =", y[1][4][3])
    print("y.shape = ", y.shape)
    print("caches[1][1[1]] =", caches[1][1][1])
    print("c[1][2][1]", c[1][2][1])
    print("len(caches) = ", len(caches))



# *_*coding:utf-8 *_*

import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5., 4.)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

def zero_pad(X, pad):
    '''

    :param X:
    :param pad:
    :return: X_paded
    '''

    X_paded = np.pad(X, (
                        (0, 0),
                        (pad, pad),
                        (pad, pad),
                        (0, 0)),
                        'constant', constant_values=0)

    return X_paded

def conv_single_step(a_slice_prev, W, b):
    '''

    :param a_slice_prev:
    :param W:
    :param b:
    :return:
    '''

    s = np.multiply(a_slice_prev, W) + b
    Z = np.sum(s)

    return Z

def conv_forward(A_prev, W, b, hparameters):
    '''

    :param A_prev:
    :param W:
    :param b:
    :param hparameters:
    :return:
    '''

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    (h_filter, f_filter, n_C_prev, n_C) = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    n_H = int((n_H_prev - h_filter + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f_filter + 2 * pad) / stride) + 1

    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + h_filter
                    horiz_start = w * stride
                    horiz_end = horiz_start + f_filter
                    a_slice_prev = a_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[0, 0, 0, c])

    assert(Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hparameters)

    return (Z, cache)

def pool_forward(A_prev, hparameters, mode='max'):
    '''

    :param A_prev:
    :param hparameters:
    :param mode:
    :return:
    '''

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    stride = hparameters['stride']
    f = hparameters['f']

    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    a_slice_prev = A_prev[i, vert_start: vert_end, horiz_start: horiz_end, c]
                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_slice_prev)
                    elif mode == 'average':
                        A[i, h, w, c] = np.mean(a_slice_prev)

    assert(A.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, hparameters)

    return (A, cache)

def conv_backward(dZ, cache):
    '''

    :param dZ:
    :param cache:
    :return: dA_prev    dW  db
    '''

    (A_prev, W, b, hparameters) = cache

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dZ.shape

    (h_filter, w_filter, n_C_prev, n_C) = W.shape

    pad = hparameters['pad']
    stride = hparameters['stride']

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros((1, 1, 1, n_C))

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + h_filter
                    horiz_start = w * stride
                    horiz_end = horiz_start + w_filter
                    a_slice_prev = a_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :]

                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice_prev * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    cache = (A_prev, W, b, hparameters)

    return (dA_prev, dW, db)

def create_mask_from_window(x):
    '''

    :param x:
    :return:
    '''

    mask = x == np.max(x)

    return mask

def distribute_value(dz, shape):
    '''

    :param dz:
    :param shape:
    :return:
    '''

    (n_H, n_W) = shape
    average = dz / (n_H * n_W)

    a = np.ones(shape) * average

    return a

def pool_backward(dA, cache, mode='max'):
    '''

    :param dA:
    :param cache:
    :param mode:
    :return:
    '''

    (A_prev, hparameters) = cache

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dA.shape

    f = hparameters['f']
    stride = hparameters['stride']

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        a_prev = A_prev[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    if mode == 'max':
                        a_slice_prev = a_prev[vert_start: vert_end, horiz_start: horiz_end, c]
                        mask = create_mask_from_window(a_slice_prev)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                    elif mode == 'max':
                        da = dA[i, h, w, c]
                        shape = (f, f)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)

    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev

if __name__ == '__main__':
    np.random.seed(1)
    # x = np.random.randn(4, 3, 3, 2)
    # x_paded = zero_pad(x, 2)
    # print(x.shape)
    # print(x_paded.shape)
    # print(x[1, 1])
    # print(x_paded[1, 1])
    #
    # fig, axarr = plt.subplots(1, 2)
    # axarr[0].set_title('X')
    # axarr[0].imshow(x[0, :, :, 0])
    # axarr[1].set_title('x_paded')
    # axarr[1].imshow(x_paded[0, :, :, 0])
    # plt.show()

    A_prev = np.random.rand(10, 4, 4, 3)
    W = np.random.randn(2, 2, 3, 8)
    b = np.random.randn(1, 1, 1, 8)
    hparameters = {'pad': 2, 'stride': 1}
    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    print(np.mean(Z))
    print(cache_conv[0][1][2][3])
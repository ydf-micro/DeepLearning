# *_*coding:utf-8 *_*
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.python.framework import ops

import cnn_utils

np.random.seed(1)

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    '''

    :param n_H0:
    :param n_W0:
    :param n_C0:
    :param n_y:
    :return:
    '''

    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])

    return X, Y

def initialize_parameters():
    '''

    :return:
    '''

    tf.set_random_seed(1)
    W1 = tf.get_variable('W1', [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W2', [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {'W1': W1,
                  'W2': W2}

    return parameters

def forward_propagation(X, parameters):
    '''

    :param X:
    :param parameters:
    :return:
    '''

    W1 = parameters['W1'] * np.sqrt(2)
    W2 = parameters['W2'] * np.sqrt(2)

    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')

    A1 = tf.nn.relu(Z1)

    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')

    A2 = tf.nn.relu(Z2)

    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

    P = tf.contrib.layers.flatten(P2)

    Z3 = tf.contrib.layers.fully_connected(P, 6, activation_fn=None)

    return Z3

def compute_cost(Z3, Y):
    '''

    :param Z3:
    :param Y:
    :return:
    '''

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate=.009,
          num_epochs=1500, minibatch_size=64, print_cost=True, isPlot=True):
    '''

    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :param learning_rate:
    :param nbum_epochs:
    :param minibatch_size:
    :param print_cost:
    :param isPlot:
    :return:
    '''

    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed += 1
            minibatches = cnn_utils.random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

            if print_cost:
                if epoch % 100 == 0:
                    print('当前是第', epoch, '代，成本值为：', minibatch_cost)


            costs.append(minibatch_cost)

        if isPlot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title('Learning rate=' + str(learning_rate))
            plt.show()

        correct_prediction = tf.equal(tf.argmax(Z3, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        train_accuracy = accuracy.eval(feed_dict={X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval(feed_dict={X: X_test, Y: Y_test})

        print('训练集准确度：', train_accuracy)
        print('测试集准确度：', test_accuracy)

if __name__ == "__main__":
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = cnn_utils.load_dataset()
    # index = 6
    # plt.imshow(X_test_orig[index])
    # print(np.squeeze(Y_train_orig[:, index]))
    # plt.show()
    X_train = X_train_orig / 255
    X_test = X_test_orig / 255
    Y_train = cnn_utils.convert_to_one_hot(Y_train_orig, 6).T
    Y_test = cnn_utils.convert_to_one_hot(Y_test_orig, 6).T

    model(X_train, Y_train, X_test, Y_test, num_epochs=1500)


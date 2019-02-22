# *_*coding:utf-8 *_*

import numpy as np
import random
import time
import cllm_utils

def clip(gradients, maxValue):
    '''

    :param gradients:
    :param maxValue:
    :return:
    '''

    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']

    for gradient in [dWaa, dWax, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients

def sample(parameters, char_to_ix, seed):
    '''

    :param parameters:
    :param char_to_is:
    :param seed:
    :return:
    '''

    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    x = np.zeros((vocab_size, 1))

    a_prev = np.zeros((n_a, 1))

    indices = []

    idx = -1

    counter = 0
    newline_character = char_to_ix['\n']

    while(idx != newline_character and counter < 50):
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)

        z = np.dot(Wya, a) + by

        y = cllm_utils.softmax(z)

        np.random.seed(counter + seed)

        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())

        indices.append(idx)

        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        a_prev = a
        seed += 1
        counter += 1
    if counter == 50:
        indices.append(char_to_ix['\n'])

    return indices

def optimizer(X, Y, a_prev, parameters, learning_rate=.01):
    '''

    :param X:
    :param Y:
    :param a_prev:
    :param parameters:
    :param learning_rate:
    :return:
    '''

    loss, cache = cllm_utils.rnn_forward(X, Y, a_prev, parameters)

    gradients, a = cllm_utils.rnn_backward(X, Y, parameters, cache)

    gradients = clip(gradients, 5)
    parameters = cllm_utils.update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X)-1]

def model(data, ix_to_char, char_to_ix, num_iterations=3500,
          n_a=50, dino_names=7, vocab_size=27):
    '''

    :param data:
    :param ix_to_char:
    :param char_to_ix:
    :param num_iterations:
    :param n_a:
    :param dino_name:
    :param vocab_size:
    :return:
    '''

    n_x, n_y = vocab_size, vocab_size

    parameters = cllm_utils.initialize_parameters(n_a, n_x, n_y)

    loss = cllm_utils.get_initial_loss(vocab_size, dino_names)

    with open('dinos.txt', 'r') as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    np.random.seed(0)
    np.random.shuffle(examples)

    a_prev = np.zeros((n_a, 1))

    for j in range(num_iterations):
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix['\n']]

        curr_loss, gradients, a_prev = optimizer(X, Y, a_prev, parameters)

        loss = cllm_utils.smooth(loss, curr_loss)

        if j % 2000 == 0:
            print('损失值为：', loss)

            seed = 0
            for _ in range(dino_names):
                sampled_indices = sample(parameters, char_to_ix, seed)
                cllm_utils.print_sample(sampled_indices, ix_to_char)

                seed += 1

            print('\n')

    return parameters

if __name__ == '__main__':
    with open('dinos.txt', 'r') as f:
        data = f.read()

    data = data.lower()

    chars = list(set(data))

    data_size, vocab_size = len(data), len(chars)
    # print(chars)
    # print(data_size, vocab_size)

    char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
    ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
    start_time = time.clock()
    parameters = model(data, ix_to_char, char_to_ix, num_iterations=3500)
    end_time = time.clock()
    minium = end_time - start_time
    print(minium)
    print(minium//60, 'm', minium%60, 's')
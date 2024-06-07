# Simple NN components, assuming 1d structure with 1 filter

import numpy as np

def relu(x):
    return np.maximum(x, 0)

def linear(x, weights):
    w_left, w_mid, w_right = weights
    return w_left * np.roll(x, 1) + w_mid * x + w_right * np.roll(x, -1)

def reclin(x, weights, bias):
    return relu(linear(x, weights) + bias)

# CNN that implements rule 110
def advance_by_cnn(x):
    temp1 = linear(x, [0, 1, 1])
    temp2 = reclin(x, [1, 1, 1], -2)
    result = temp1 - 2 * temp2
    # The result is almost right, but sometimes 2 appears instead of 1
    result = result - relu(result - 1)
    return result

# Run simulation with a single full cell

def initial_state():
    return np.eye(50)[40]

def draw_state(state):
    chars = {0.0: '.', 1.0: '█'}
    print(''.join(chars[x] for x in state))

state = initial_state()
for i in range(30):
    draw_state(state)
    state = advance_by_cnn(state)

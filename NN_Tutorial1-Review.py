import numpy as np

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def deri_sigmoid(x):
    return x * (1-x)

training_input = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
training_output = np.array([[0], [1], [1], [0]])

weights = 2 * np.random.random((3,1)) - 1

print('Random Starting Synaptic Weights: ', weights)

for iteration in range(10000):
    input_layer  = training_input
    output = sigmoid(np.dot(input_layer, weights))
    error = training_output - output
    adjust = error * deri_sigmoid(output)
    weights += np.dot(input_layer.T, adjust)

print("synaptic weights after training", weights)
print("outputs after training", output)
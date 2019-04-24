import numpy as np

def sigmoid(x):
    return 1 / (1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

training_input = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
training_output = np.array([[0], [1], [1], [0]])

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1)) - 1
print('Random Starting Synaptic Weights: ', synaptic_weights)

for iteration in range(50000):
    input_layer =  training_input
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    error = training_output - outputs
    adjustment = error * sigmoid_derivative(outputs)
    synaptic_weights += np.dot(input_layer.T, adjustment)

print("synaptic weights after training", synaptic_weights)
print("outputs after training", outputs)

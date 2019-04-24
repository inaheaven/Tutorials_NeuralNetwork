import numpy as np

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        self.weight = 2 * np.random.random((3,1)) - 1
        self.output = np.empty((4,1))

    def forward(self, x):
        return 1 / (1+np.exp(-x))

    def backward(self, x):
        return x * (1-x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            input_layer = training_inputs
            self.output = self.forward(np.dot(input_layer, self.weight))
            error = training_outputs - self.output
            adjust = error * self.backward(self.output)
            self.weight += np.dot(input_layer.T, adjust)



if __name__ == "__main__":
    neural_net = NeuralNetwork()
    print("Initial Weights: ", neural_net.weight)

    training_input = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_output = np.array([[0], [1], [1], [0]])

    neural_net.train(training_input, training_output, 50000)
    print("outputs after training", neural_net.weight)
    print("outputs after training", neural_net.output)

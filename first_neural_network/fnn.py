import numpy as np


class Neural_Network(object):

    def __init__(self):
        '''
        declare hyperparameters to define structure
        of network
        all constant values
        '''
        self.input_layer_size = 2
        self.output_layer_size = 1
        self.hidden_layer_size = 3

        # initialize weight matrices to random values
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size)

    def forward(self, X):
        '''pass multiple inputs at once
        through Neural_Network using matrices.
        Compute yHat or actual cost.'''
        # find dot product of input matrix
        # and weight matrix for first layer.
        self.z2 = np.dot(X, self.W1)
        # apply sigmoid to matrix
        self.a2 = self.sigmoid(self.z2)
        # multiply output by second layer of weights
        self.z3 = np.dot(self.a2, self.W2)
        # apply sigmoid again
        yHat = self.sigmoid(self.z3)

        return yHat

    # declare as static if you are not using the object
    # (self) inside the method.
    # i.e.you really don't care about the object
    # that your method is bound to.
    @staticmethod
    def sigmoid(z):
        '''apply sigmoid activation
        pass in a scalar/vector/matrix
        using numpy, numpy applies sigmoid
        to each element in the vector'''
        return 1 / (1 + np.exp(-z))


def log_it(text):
    print(text)


def main():
    # Matrix of inputs
    X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
    # print(X)

    # matrix of (expected) outputs for the inputs above.
    y = np.array(([75], [82], [93]), dtype=float)
    # print(y)

    '''Normalize values and bring them between 0 and 1, so as to
    make them comparable'''
    # divides each x1 value by highest x1 and each x2 value by highest x2
    X = X / (np.amax(X, axis=0))
    # do the same here
    y = y / 100
    # print(y / 100)

    fnn = Neural_Network()

    yHat = fnn.forward(X)

    print(yHat)

if __name__ == "__main__":
    main()

import numpy as np
import math
import random


# ============================================================
# Linear / Fully Connected Layer
# ============================================================

class Linear:
    def __init__(self, input_data, no_of_perceptrons):
        # Each column of the weight matrix represents one perceptron
        self.weights = np.random.randn(input_data, no_of_perceptrons) * 0.1

        # One bias for each perceptron
        self.bias = np.zeros((1, no_of_perceptrons))

        # Used to identify layers with trainable parameters
        self.trainable = True

    def forward(self, input_data):
        # Store input for calculating the weight gradient
        self.input_data = input_data

        # Z = XW + b
        self.out = self.input_data @ self.weights + self.bias

        return self.out

    def backward(self, gradient):
        # Gradient of loss with respect to weights
        self.gradient_weights = self.input_data.T @ gradient

        # Gradient of loss with respect to biases
        self.gradient_bias = gradient.sum(axis=0, keepdims=True)

        # Gradient passed to the previous layer
        self.gradient_input_data = gradient @ self.weights.T

        return self.gradient_input_data


# ============================================================
# Sigmoid Activation
# ============================================================

class sigmoid:
    def __init__(self):
        self.trainable = False

    def forward(self, input_data):
        # Sigmoid: 1 / (1 + e^(-x))
        self.out = 1 / (1 + np.exp(-input_data))

        return self.out

    def backward(self, gradient):
        # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        return gradient * self.out * (1 - self.out)


# ============================================================
# ReLU Activation
# ============================================================

class relu:
    def __init__(self):
        self.trainable = False

    def forward(self, input_data):
        # ReLU(x) = max(0, x)
        self.out = np.maximum(0, input_data)

        return self.out

    def backward(self, gradient):
        # ReLU derivative is 1 for positive inputs and 0 otherwise
        return gradient * (self.out > 0)


# ============================================================
# Mean Squared Loss
# ============================================================

class mean_squared_loss:
    def __init__(self):
        self.trainable = False

    def forward(self, y_pred, y):
        # Store values required during backpropagation
        self.y_pred = y_pred
        self.y = y

        # L = (1 / 2N) * sum((y_pred - y)^2)
        self.loss = ((self.y_pred - self.y) ** 2).sum() / (2 * y.shape[0])

        return self.loss

    def backward(self):
        # dL/dy_pred
        return (self.y_pred - self.y) / self.y.shape[0]


# ============================================================
# Neural Network Model
# ============================================================

class model:
    def __init__(self, layers, loss_function):
        # Layers are executed in the order given here
        self.layers = layers
        self.loss_function = loss_function

    def forward(self, input_data):
        # Pass the output of each layer to the next layer
        output = input_data

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, gradient):
        # Backpropagate from the last layer to the first
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def train(self, x, y, learning_rate=0.001, epochs=100000):
        # Used for the early-stopping condition
        previous_loss = float("inf")

        for epoch in range(epochs):

            # Forward propagation
            out = self.forward(x)

            # Calculate loss
            loss = self.loss_function.forward(out, y)

            # Backward propagation
            gradient = self.loss_function.backward()
            self.backward(gradient)

            # Update trainable parameters
            for layer in self.layers:
                if layer.trainable == True:
                    layer.weights = (
                        layer.weights
                        - learning_rate * layer.gradient_weights
                    )

                    layer.bias = (
                        layer.bias
                        - learning_rate * layer.gradient_bias
                    )

            # Display training progress
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss}")

            # Early stopping
            if abs(previous_loss - loss) < 0.0000001:
                break

            previous_loss = loss

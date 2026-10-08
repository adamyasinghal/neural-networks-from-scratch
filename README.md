# Neural Network From Scratch

A neural network framework built from scratch using **Python and NumPy**, without using deep-learning frameworks such as PyTorch or TensorFlow.

The goal of this project is to understand the mathematics and mechanics behind neural networks by implementing the core components manually.

## Current Features

- Fully connected (`Linear`) layers
- Multiple perceptrons per layer
- Arbitrary stacking of layers
- Forward propagation
- Backpropagation
- Weight and bias gradients
- ReLU activation
- Sigmoid activation
- Mean squared loss
- Full-batch gradient descent
- Basic early stopping
- Trainable/non-trainable layer distinction

## Current Architecture

The framework currently supports networks such as:

```text
Input
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
Sigmoid
  ↓
Linear
  ↓
Output
```

Layers can be stacked in different configurations without modifying the `model` class.

## Example

The current example trains a multilayer neural network to approximate:

```text
y = (x + 1)²
```

The example is contained separately in `example.py`.

Run it with:

```bash
python example.py
```

## Project Structure

```text
neural-network-from-scratch/
│
├── neural_network.py    # Neural network framework
├── example.py           # Example usage
├── README.md
└── requirements.txt
```

## Implementation

The framework uses NumPy for matrix operations and implements the forward and backward passes manually.

For a linear layer:

```text
Z = XW + b
```

During backpropagation:

```text
dW = Xᵀ · dZ

db = ΣdZ

dX = dZ · Wᵀ
```

The network uses full-batch training, meaning the complete training dataset is passed through the network during each training step.

## Roadmap

This project will gradually expand into a more complete deep-learning framework.

### Completed

- [x] Linear / fully connected layers
- [x] Multiple perceptrons per layer
- [x] Multiple stacked layers
- [x] Forward propagation
- [x] Backpropagation
- [x] ReLU
- [x] Sigmoid
- [x] Mean squared loss
- [x] Full-batch gradient descent
- [x] Basic early stopping

### Planned

- [ ] Dropout
- [ ] Batch Normalization
- [ ] Additional activation functions
- [ ] Additional loss functions
- [ ] Additional optimizers
- [ ] Gradient checking
- [ ] Convolutional layers
- [ ] Pooling layers
- [ ] CNNs
- [ ] More model utilities

## Purpose

This is primarily an **educational project**.

The objective is to understand what happens inside neural-network frameworks by implementing the underlying mathematics and algorithms from scratch.

Instead of immediately relying on PyTorch or TensorFlow, the project explores how:

- Layers transform data
- Forward propagation works
- Gradients are calculated
- Backpropagation propagates gradients
- Parameters are updated
- Different neural-network components interact

## Requirements

- Python 3
- NumPy

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## License

This project is intended primarily for learning and experimentation.

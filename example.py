import numpy as np

from neural_network import (
    Linear,
    relu,
    sigmoid,
    mean_squared_loss,
    model
)


# ============================================================
# Generate training data
# ============================================================

X = np.linspace(-5, 5, 100).reshape(-1, 1)

# Target function: y = (x + 1)^2
y = (X + 1) ** 2


# ============================================================
# Create model
# ============================================================

network = model(
    [
        Linear(1, 10),
        relu(),
        Linear(10, 10),
        relu(),
        Linear(10, 10),
        sigmoid(),
        Linear(10, 1)
    ],
    mean_squared_loss()
)


# ============================================================
# Train
# ============================================================

network.train(
    X,
    y
)


# ============================================================
# Test predictions
# ============================================================

predictions = network.forward(X)

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y)

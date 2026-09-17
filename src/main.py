import numpy as np

X = np.array([
    [1, 1],
    [2, 1],
    [4, 3],
    [5, 4]
], dtype=np.float32)

y = np.array([0, 0, 1, 1], dtype=np.float32)

w = np.zeros(2, dtype=np.float32)
b = 0.0

learning_rate = 0.1
epochs = 1000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


for epoch in range(epochs):
    z = X @ w + b
    p = sigmoid(z)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    dw = X.T @ (p - y) / len(y)
    db = np.mean(p - y)

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}: "
            f"loss={loss:.6f}, "
            f"w={w}, "
            f"b={b:.6f}"
        )

    w -= learning_rate * dw
    b -= learning_rate * db
print("Probabilities:", sigmoid(X @ w + b))
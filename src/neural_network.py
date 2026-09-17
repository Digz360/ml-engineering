import numpy as np


class DenseLayer:
    def __init__(self, input_size, output_size):
        self.W = (
                    np.random.randn(input_size, output_size)
                    * np.sqrt(1 / input_size)
                )
        self.b = np.zeros((1, output_size))

        self.mW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)

        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

        self.t = 0

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.b
        return self.Z

    def backward(self, dZ):
        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)
        self.dX = dZ @ self.W.T

        return self.dX

    def update(self, learning_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):

        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.mb = beta1 * self.mb + (1 - beta1) * self.db

        self.vW = beta2 * self.vW + (1 - beta2) * self.dW ** 2
        self.vb = beta2 * self.vb + (1 - beta2) * self.db ** 2

        self.t += 1

        mW_corrected = self.mW / (1 - beta1 ** self.t)
        mb_corrected = self.mb / (1 - beta1 ** self.t)
        vW_corrected = self.vW / (1 - beta2 ** self.t)
        vb_corrected = self.vb / (1 - beta2 ** self.t)

        self.W -= learning_rate * mW_corrected / (np.sqrt(vW_corrected) + epsilon) 
        self.b -= learning_rate * mb_corrected / (np.sqrt(vb_corrected) + epsilon) 


class ReLU:
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        dZ = dA * (self.Z > 0)
        return dZ

class Tanh:
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dA):
        dZ = dA * (1 - self.A ** 2)
        return dZ

class Sigmoid:
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, dA):
        return dA * self.A * (1 - self.A)


class BinaryCrossEntropy:
    def forward(self, Y_hat, Y):
        eps = 1e-8

        loss = -np.mean(
            Y * np.log(Y_hat + eps)
            + (1 - Y) * np.log(1 - Y_hat + eps)
        )

        return loss

    def backward(self, Y_hat, Y):
        N = Y.shape[0]
        return (Y_hat - Y) / N

if __name__ == "__main__":

    np.random.seed(42)

    dense1 = DenseLayer(2, 4)
    activation = Tanh()

    dense2 = DenseLayer(4, 1)
    sigmoid = Sigmoid()

    loss = BinaryCrossEntropy()

    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])

    Y = np.array([
        [0.0],
        [0.0],
        [0.0],
        [1.0]
    ])

    learning_rate = 0.5

    for epoch in range(100):

        # ---------- FORWARD ----------
        Z1 = dense1.forward(X)
        A1 = activation.forward(Z1)

        Z2 = dense2.forward(A1)
        Y_hat = sigmoid.forward(Z2)

        L = loss.forward(Y_hat, Y)

        # ---------- BACKWARD ----------
        dZ2 = loss.backward(Y_hat, Y)

        dA1 = dense2.backward(dZ2)
        dZ1 = activation.backward(dA1)
        dense1.backward(dZ1)

        # ---------- UPDATE ----------
        dense1.update(learning_rate)
        dense2.update(learning_rate)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {L:.4f}")



    print("\nPredictions:")
    print(Y_hat)

    print("\nClasses:")
    print((Y_hat > 0.5).astype(int))

    print("\nW1:")
    print(dense1.W)

    print("\nb1:")
    print(dense1.b)

    print("\nW2:")
    print(dense2.W)

    print("\nb2:")
    print(dense2.b)
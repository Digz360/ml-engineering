import numpy as np
import matplotlib.pyplot as plt
from neural_network import DenseLayer, Tanh, Sigmoid, BinaryCrossEntropy

np.random.seed(42)

X = np.random.uniform(-1.5, 1.5, size=(2000, 2))

Y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 1).astype(float)
Y = Y.reshape(-1, 1)


X_train = X[:1600]
Y_train = Y[:1600]

X_test = X[1600:]
Y_test = Y[1600:]


layer1 = DenseLayer(2, 8)
activation1 = Tanh()

layer2 = DenseLayer(8, 1)
sigmoid = Sigmoid()

loss_fn = BinaryCrossEntropy()



for epoch in range(20000):

    #FORWARD

    Z1 = layer1.forward(X_train)

    A1 = activation1.forward(Z1)

    Z2 = layer2.forward(A1)

    A2 = sigmoid.forward(Z2)

    L = loss_fn.forward(A2, Y_train)

    #BACKWARD

    dZ2 = loss_fn.backward(A2, Y_train)

    dA1 = layer2.backward(dZ2) 

    dZ1 = activation1.backward(dA1)

    layer1.backward(dZ1)

    #UPDATE

    layer1.update(0.1)
    layer2.update(0.1)


    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {L:.4f}")

Z1_test = layer1.forward(X_test)
A1_test = activation1.forward(Z1_test)

Z2_test = layer2.forward(A1_test)
A2_test = sigmoid.forward(Z2_test)


predictions = (A2_test >= 0.5).astype(int)

accuracy = np.mean(predictions == Y_test)

print(f"Test accuracy: {accuracy * 100:.2f}%")



x1 = np.linspace(-1.5, 1.5, 200)
x2 = np.linspace(-1.5, 1.5, 200)

xx1, xx2 = np.meshgrid(x1, x2)

grid = np.c_[xx1.ravel(), xx2.ravel()]

# Forward nos pontos da grelha
Z1_grid = layer1.forward(grid)
A1_grid = activation1.forward(Z1_grid)

Z2_grid = layer2.forward(A1_grid)
A2_grid = sigmoid.forward(Z2_grid)

# Voltar à forma 200x200
decision = A2_grid.reshape(xx1.shape)

# Desenhar
plt.contourf(xx1, xx2, decision, levels=20)

plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=Y_test[:, 0],
    edgecolors="black"
)

plt.contour(
    xx1,
    xx2,
    decision,
    levels=[0.5]
)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Decision Boundary aprendida pela Neural Network")
plt.show()



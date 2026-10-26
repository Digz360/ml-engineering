import numpy as np
import torch


#AMBIENTE

X = torch.tensor([
    [1., 2.],
    [3., 4.]
])

W = torch.tensor([
    [0.5, -0.2],
    [0.1,  0.3]
], requires_grad=True)

b = torch.tensor([[0., 0.]], requires_grad=True)


#FORWARD
Z = X @ W + b
optimizer = torch.optim.SGD([W, b], lr=0.01)
loss = Z.sum()
optimizer.zero_grad()

#BACKWARD
loss.backward()
optimizer.step() #ATUALIZAR

x = torch.tensor([1., 2., 3.], requires_grad=True)

a = x * 2
b = a.detach().clone()

b[0] = 100

print("a:", a)
print("b:", b)
import math
class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(children)
        self._op = op
        self._backward = lambda: None

    def __mul__(self, other):
        out = Value(
            self.data * other.data,
            (self, other),
            '*'
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __add__(self, other):
        out = Value(
            self.data + other.data,
            (self, other),
            '+'
        )

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out    

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)

                topo.append(v)

        build_topo(self)    

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()    

    def tanh(self):
        t = math.tanh(self.data)

        out = Value(
            t,
            (self,),
            'tanh'
        )

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward

        return out

    def __neg__(self):
        return self * Value(-1.0)

    def __sub__(self, other):
        return self + (-other)    




x1 = Value(2.0)
x2 = Value(-1.0)

w1 = Value(0.5)
w2 = Value(-0.3)
b = Value(0.1)

y = Value(1.0)

a = (x1 * w1 + x2 * w2 + b).tanh()

error = a - y
loss = error * error

loss.backward()

learning_rate = 0.1

w1.data -= learning_rate * w1.grad
w2.data -= learning_rate * w2.grad
b.data -= learning_rate * b.grad

a2 = (x1 * w1 + x2 * w2 + b).tanh()

error2 = a2 - y
loss2 = error2 * error2

w1.grad = 0.0
w2.grad = 0.0
b.grad = 0.0

loss2.backward()



print("--------------------------")


print("prediction:", a.data)
print("loss:", loss.data)

print("w1 grad:", w1.grad)
print("w2 grad:", w2.grad)
print("b grad:", b.grad)

print("Old loss:", loss.data)
print("New loss:", loss2.data)


print("--------------------------")

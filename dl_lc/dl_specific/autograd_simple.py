"""
Autograd for scalar values.
"""
from typing import Union


class Scalar:
    def __init__(self, data, _children=()):
        assert isinstance(data, int) or isinstance(data, float), (f"Data format"
                                                                  f"not supported: {type(data)}")
        self.data = data

        # Backprop metadata.
        self._grad = 0
        self._children = set(_children)
        self._grad_op = lambda: None

    def _wrap_int_or_float(self, x):
        if not isinstance(x, Scalar):
            x = Scalar(x)
        return x

    def __add__(self, other):
        def _add_backward():
            # dself = dout * Jself
            # dother = dout * Jother
            self._grad += out._grad
            other._grad += out._grad

        other = self._wrap_int_or_float(other)

        # Create output instance.
        out = Scalar(self.data + other.data, _children=(self, other))

        # Set gradient op for output instance.
        out._grad_op = _add_backward

        return out

    def __mul__(self, other):
        def _mul_backward():
            self._grad += out._grad * other.data
            other._grad += out._grad * self.data

        other = self._wrap_int_or_float(other)

        out = Scalar(self.data * other.data, _children=(self, other))

        out._grad_op = _mul_backward

        return out

    def __pow__(self, n):
        def _pow_backward():
            self._grad += out._grad * (n * (self.data ** (n - 1)))

        out = Scalar(self.data ** n, _children=(self,))

        out._grad_op = _pow_backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def zero_grad(self):
        self._grad = 0
        for child in self._children:
            child.zero_grad()

    def backward(self):
        # `self` is root node.
        self._grad = 1
        
        # Build DAG traversal order.
        order = []
        _visited = set()
        def _traverse(node):
            # Visit node.
            order.append(node)
            _visited.add(node)
            
            # Recurse on children.
            for child in node._children:
                if child not in _visited:
                    _traverse(child)
        _traverse(self)

        # Call backward starting from root.
        for node in order:
            node._grad_op()


def main():
    # Scalar linear regression.
    x = 8
    w_true = 6
    b_true = 2
    y = x * w_true + b_true

    lr = 0.01
    w = Scalar(1)
    b = Scalar(1)
    for i in range(100):
        pred = w * x + b
        loss = (pred - y) ** 2
        loss.backward()
        w = w - lr * w._grad
        b = b - lr * b._grad
        loss.zero_grad()

        print("*" * 50)
        print(f"Iteration: {i}")
        print(f"loss: {loss.data}")
        print(f"w_true: {w_true}, w: {w.data}")
        print(f"b_true: {b_true}, b: {b.data}")


if __name__ == "__main__":
    main()

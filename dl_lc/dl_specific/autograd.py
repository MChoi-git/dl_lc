from typing import Dict, List
from functools import partial
from contextlib import contextmanager
import numpy as np

from jax import numpy as jnp    # For testing grads


# NOTE: Input vectors are row-major


class Tensor:
    """Edges in the graph. Saves the FunctionNode used to create it."""
    def __init__(self, ndarray, prev_op=None):
        self.ndarray = ndarray

        # Backward meta.
        self.grad = None
        self.prev_op = prev_op

    def shape(self):
        return self.ndarray.shape

    def __repr__(self):
        return (f"Tensor(shape={self.ndarray.shape}, grad={self.grad}, "
                f"prev_op={self.prev_op})")


class FunctionNode:
    """
    Function node in the computation graph. Contains metadata for running
    backprop.
    """
    def __init__(self, name, grad_fn, children, _inputs):
        self.name = name

        # Backward meta.
        self.grad_fn = grad_fn
        self.children = children
        self._inputs = _inputs

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return (f"GradNode{self.name}(grad_fn={self.grad_fn}, "
                f"children={self.children}, _inputs={self._inputs})")


def VADD(x1: Tensor, x2: Tensor):
    """Adds two vectors elementwise."""
    out = Tensor(x1.ndarray + x2.ndarray)

    def backward(dL):
        """dx1 = dL * I and dx2 = dL * I."""
        dx1 = Tensor(dL.ndarray * np.eye(dL.shape()[0]))
        dx2 = Tensor(dL.ndarray * np.eye(dL.shape()[0]))
        return dx1, dx2

    return out, backward


def VMUL(x1: Tensor, x2: Tensor):
    """Multiplies two vectors elementwise."""
    out = Tensor(x1.ndarray * x2.ndarray)

    def backward(dL):
        """dx1 = dL @ x2.T and dx2 = dL @ x1.T."""
        dx1 = Tensor(dL.ndarray * np.eye(dL.shape()[0]) * x2.ndarray)
        dx2 = Tensor(dL.ndarray * np.eye(dL.shape()[0]) * x1.ndarray)
        return dx1, dx2

    return out, backward


def VMADD(x1, x2):
    """Vector multiply-add (vector multiplication)."""
    out = x1 @ x2.T

    def backward( L):
        dx1 = dL @ x2.T
        dx2 = dL @ x1.T
        return dx1, dx2

    return out, backward

    
def SUM(x):
    """Sum-reduce a vector."""
    out = x.sum()

    def backward(dL):
        dx = dL * np.ones(dL.shape[0]).T
        return dx

    return out, backward


class Graph:
    def __init__(self):
        self.graph = {} # Dict{fn_node.name: Tuple(fn_node, Tensors, Tensors)}
        self.node_count = 0
        self.last_recorded_node = None

    def clear_graph(self):
        self.graph = {}

    def _visit(self, node, dL, visited):
        """Run grad function for node, then deposit grads into inputs tensors."""
        grads = node.grad_fn(dL)
        for g, t in zip(grads, node._inputs):
            t.grad = g

        visited.append(node)

    def _dfs(self, node, visited, dL):
        self._visit(node, dL, visited)

        for input_ten in node._inputs:
            if input_ten.prev_op is not None:
                next_node = input_ten.prev_op
                dL = input_ten.grad
                self._dfs(next_node, visited, dL)


    def traverse(self, root, dL):
        # TODO: Implement backward traversal.
        # NOTE: See https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py.
        #       Does autograd using DFS topological sort
        visited = []
        root_node = self.graph[root]

        self._dfs(root_node, visited, dL)


    @staticmethod
    def get_unique_fn_idx(fn):
        # For unique function naming.
        if not hasattr(fn, "idx"):
            fn.idx = 0

        idx = fn.idx
        fn.idx += 1
        return idx

    def _op(self, fn_name, fn, *args):
        idx = self.get_unique_fn_idx(fn)
        node_name = f"{fn_name}{idx}"

        children = set() # Children function nodes
        inputs = []     # Input tensors to deposit grads in.
        for arg in args:
            if isinstance(arg, Tensor):
                inputs.append(arg)

                if arg.prev_op is not None:
                    children.add(arg.prev_op)

        out, grad_fn = fn(*args)

        fn_node = FunctionNode(node_name, grad_fn, children, inputs)
        self.node_count += 1
        self.last_recorded_node = node_name

        out.prev_op = fn_node
        self.graph[node_name] = fn_node
        return out

    def vadd(self, x1: Tensor, x2: Tensor):
        out = self._op("VADD", VADD, x1, x2)
        return out

    def vmul(self, x1: Tensor, x2: Tensor):
        out = self._op("VMUL", VMUL, x1, x2)
        return out

    def vmadd(self, x1: Tensor, x2: Tensor):
        out = self._op("VMADD", VMADD, x1, x2)
        return out

    def sum(self, x: Tensor):
        out = self._op("SUM", SUM, x)
        return out

    def backward(self, dL):
        root = self.last_recorded_node
        order = self.traverse(root, dL)


def main():
    np.random.seed(0)
    x = np.random.randn(100, 64)
    w_true = np.random.randn(64)
    b_true = np.random.randn(64)
    noise = np.random.normal(scale=0.1, size=(100, 64))
    data = x + noise
    targets = data * w_true + b_true

    g = Graph()
    w = Tensor(np.random.randn(64,))
    b = Tensor(np.random.randn(64,))
    for example, target in zip(data, targets):
        g.clear_graph()
        example = Tensor(example)
        target = Tensor(-target)

        vmul_out = g.vmul(example, w)
        vadd_out = g.vadd(vmul_out, b)
        diff = g.vadd(vadd_out, target)
        g.backward(diff)
        print(w.grad)
        print(b.grad)
        print(vmul_out.grad)
        print(vadd_out.grad)
        print(diff.grad)
        breakpoint()


if __name__ == "__main__":
    main()

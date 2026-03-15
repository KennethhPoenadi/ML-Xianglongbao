# Implementasi automatic differntiation (dengan computation graph)

from __future__ import annotations
from typing import List, Set, Tuple, Union
import numpy as np

def _reduce_broadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    if grad.shape == shape:
        return grad
    if shape == ():
        return np.array(grad.sum())
    for _ in range(grad.ndim - len(shape)):
        grad = grad.sum(axis=0)
    for i in range(len(shape)):
        if shape[i] == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    # Tensor dengan dukungan automatic differentiation.

    def __init__(
        self,
        data: Union[np.ndarray, list, float, int],
        _children: tuple = (),
        _op: str = "",
        requires_grad: bool = False,
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None 
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def T(self) -> Tensor:
        out = Tensor(self.data.T, (self,), "T")

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, op='{self._op}')"

    def __matmul__(self, other: Tensor) -> Tensor:
        # Perkalian matriks: C = A @ B
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T  
            other.grad += self.data.T @ out.grad 

        out._backward = _backward
        return out

    def __add__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _reduce_broadcast(out.grad, self.shape)
            other.grad += _reduce_broadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _reduce_broadcast(out.grad * other.data, self.shape)
            other.grad += _reduce_broadcast(out.grad * self.data, other.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self) -> Tensor:
        return self * (-1.0)

    def __sub__(self, other) -> Tensor:
        return self + (-other)

    def __rsub__(self, other) -> Tensor:
        return (-self) + other

    def __pow__(self, exp: Union[int, float]) -> Tensor:
        out = Tensor(self.data**exp, (self,), f"**{exp}")

        def _backward():
            self.grad += out.grad * exp * (self.data ** (exp - 1))

        out._backward = _backward
        return out

    def __truediv__(self, other) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** (-1.0))

    def __rtruediv__(self, other) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other * (self ** (-1.0))

    def relu(self) -> Tensor:
        out = Tensor(np.maximum(0, self.data), (self,), "relu")

        def _backward():
            self.grad += out.grad * (self.data > 0).astype(np.float64)

        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(s, (self,), "sigmoid")

        def _backward():
            self.grad += out.grad * s * (1.0 - s)

        out._backward = _backward
        return out

    def tanh_act(self) -> Tensor:
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += out.grad * (1.0 - t**2)

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        safe = np.clip(self.data, 1e-12, None)
        out = Tensor(np.log(safe), (self,), "log")

        def _backward():
            self.grad += out.grad / safe

        out._backward = _backward
        return out

    def clip(self, a_min: float, a_max: float) -> Tensor:
        out = Tensor(np.clip(self.data, a_min, a_max), (self,), "clip")

        def _backward():
            mask = ((self.data >= a_min) & (self.data <= a_max)).astype(
                np.float64
            )
            self.grad += out.grad * mask

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims: bool = False) -> Tensor:
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum"
        )

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis=axis)
            self.grad += np.broadcast_to(g, self.shape).copy()

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> Tensor:
        if axis is None:
            n = self.data.size
        elif isinstance(axis, int):
            n = self.data.shape[axis]
        else:
            n = int(np.prod([self.data.shape[a] for a in axis]))
        out = Tensor(
            self.data.mean(axis=axis, keepdims=keepdims), (self,), "mean"
        )

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis=axis)
            self.grad += np.broadcast_to(g, self.shape).copy() / n

        out._backward = _backward
        return out

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    def backward(self) -> None:
        # Hitung gradien via reverse-mode autodiff
        topo: List[Tensor] = []
        visited: Set[int] = set()

        def _build_topo(v: Tensor) -> None:
            vid = id(v)
            if vid not in visited:
                visited.add(vid)
                for child in v._prev:
                    _build_topo(child)
                topo.append(v)

        _build_topo(self)

        self.grad = np.ones_like(self.data)

        for v in reversed(topo):
            v._backward()

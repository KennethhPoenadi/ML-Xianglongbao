from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from activation import Linear, ReLU, Sigmoid, Tanh, Softmax

_ACTIVATION_MAP: Dict[str, Any] = {
    "linear":  Linear,
    "relu":    ReLU,
    "sigmoid": Sigmoid,
    "tanh":    Tanh,
    "softmax": Softmax,
}


def get_activation(name: str):
    key = name.lower()
    if key not in _ACTIVATION_MAP:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Available: {list(_ACTIVATION_MAP.keys())}"
        )
    return _ACTIVATION_MAP[key]()


class Layer(ABC):
    autograd: bool = False

    @abstractmethod
    def forward(self, x, training: bool = True): ...

    @abstractmethod
    def backward(self, grad, pre_activation: bool = False): ...

    def zero_grad(self) -> None:
        """Reset autograd gradients (override in subclass)."""
        pass

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {}

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        pass

    def get_params(self) -> Dict[str, np.ndarray]:
        return {}

    def get_grads(self) -> Dict[str, np.ndarray]:
        return {}

    def get_config(self) -> Dict[str, Any]:
        return {}


class Dense(Layer):

    def __init__(
        self,
        units: int,
        activation: str = "linear",
        l1: float = 0.0,
        l2: float = 0.0,
        init: str = "auto",
        init_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if units < 1:
            raise ValueError(f"units must be >= 1, got {units}")

        self.units: int = units
        self.activation_name: str = activation.lower()
        self.activation = get_activation(activation)
        self.l1: float = l1
        self.l2: float = l2
        self.init: str = init.lower()
        self.init_params: Dict[str, Any] = init_params or {}

        self.autograd: bool = False

        self.W = None
        self.b = None

        self._input: Optional[np.ndarray] = None
        self._z: Optional[np.ndarray] = None

        self._dW: Optional[np.ndarray] = None
        self._db: Optional[np.ndarray] = None

    def build(self, input_size: int) -> None:
        method = self.init
        p = self.init_params
        seed = p.get("seed", None)
        rng = np.random.default_rng(seed)

        if method == "zeros":
            W = np.zeros((input_size, self.units))
        elif method == "uniform":
            low = float(p.get("low", -0.1))
            high = float(p.get("high", 0.1))
            W = rng.uniform(low, high, size=(input_size, self.units))
        elif method == "normal":
            mean = float(p.get("mean", 0.0))
            std = float(p.get("std", 0.01))
            W = rng.normal(mean, std, size=(input_size, self.units))
        elif method == "xavier":
            #glorot: std = sqrt(1/fan_in)
            std = np.sqrt(1.0 / input_size)
            W = rng.normal(0.0, std, size=(input_size, self.units))
        elif method == "he":
            #he: std = sqrt(2/fan_in)
            std = np.sqrt(2.0 / input_size)
            W = rng.normal(0.0, std, size=(input_size, self.units))
        else:
            #auto: pilih he buat relu, xavier buat yang lain
            if self.activation_name == "relu":
                std = np.sqrt(2.0 / input_size)
            else:
                std = np.sqrt(1.0 / input_size)
            W = np.random.randn(input_size, self.units) * std

        if self.autograd:
            from autograd import Tensor
            self.W = Tensor(W, requires_grad=True)
            self.b = Tensor(np.zeros((1, self.units)), requires_grad=True)
        else:
            self.W = W
            self.b = np.zeros((1, self.units))

    def forward(self, x, training: bool = True):
        if self.W is None:
            self.build(x.shape[1])

        if self.autograd:
            z = x @ self.W + self.b
            act = self.activation_name
            if act == "relu":    return z.relu()
            if act == "sigmoid": return z.sigmoid()
            if act == "tanh":    return z.tanh_act()
            return z  # linear / softmax

        self._input = x
        self._z = x @ self.W + self.b
        return self.activation(self._z)

    def backward(self, grad, pre_activation: bool = False):
        if self.autograd:
            return grad  # no-op — gradients computed by autograd engine
        if pre_activation:
            dz = grad
        else:
            dz = grad * self.activation.derivative(self._z)

        #gradient regularisasi
        reg_grad = self.l1 * np.sign(self.W) + self.l2 * self.W

        self._dW = self._input.T @ dz + reg_grad
        self._db = np.sum(dz, axis=0, keepdims=True)
        dx = dz @ self.W.T
        return dx

    def zero_grad(self) -> None:
        if self.autograd and self.W is not None:
            self.W.zero_grad()
            self.b.zero_grad()

    def get_params(self) -> Dict[str, np.ndarray]:
        if self.W is None:
            return {}
        if self.autograd:
            return {"W": self.W.data, "b": self.b.data}
        return {"W": self.W, "b": self.b}

    def get_grads(self) -> Dict[str, np.ndarray]:
        if self.autograd and self.W is not None:
            return {"W": self.W.grad, "b": self.b.grad}
        if self._dW is None:
            return {}
        return {"W": self._dW, "b": self._db}

    def get_weights(self) -> Dict[str, np.ndarray]:
        if self.W is None:
            return {}
        if self.autograd:
            return {"W": self.W.data, "b": self.b.data}
        return {"W": self.W, "b": self.b}

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        if self.autograd:
            from autograd import Tensor
            self.W = Tensor(weights["W"], requires_grad=True)
            self.b = Tensor(weights["b"], requires_grad=True)
        else:
            self.W = weights["W"]
            self.b = weights["b"]

    def get_config(self) -> Dict[str, Any]:
        return {
            "type":        "Dense",
            "units":       self.units,
            "activation":  self.activation_name,
            "l1":          self.l1,
            "l2":          self.l2,
            "init":        self.init,
            "init_params": self.init_params,
        }

    def __repr__(self) -> str:
        if self.W is not None:
            s = self.W.data.shape if self.autograd else self.W.shape
            built = f"W{s}"
        else:
            built = "unbuilt"
        mode = ", autograd" if self.autograd else ""
        return (
            f"Dense(units={self.units}, activation='{self.activation_name}', "
            f"l1={self.l1}, l2={self.l2}, {built}{mode})"
        )


class RMSNorm(Layer):

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps: float = eps
        self.autograd: bool = False
        self.gamma = None
        self._input: Optional[np.ndarray] = None
        self._rms: Optional[np.ndarray] = None
        self._x_norm: Optional[np.ndarray] = None
        self._dgamma: Optional[np.ndarray] = None

    def build(self, features: int) -> None:
        if self.autograd:
            from autograd import Tensor
            self.gamma = Tensor(np.ones((features,)), requires_grad=True)
        else:
            self.gamma = np.ones((features,))
        self._dgamma = np.zeros((features,))

    def forward(self, x, training: bool = True):
        if self.gamma is None:
            self.build(x.shape[-1])

        if self.autograd:
            rms = ((x ** 2).mean(axis=-1, keepdims=True) + self.eps) ** 0.5
            return self.gamma * (x / rms)

        self._input = x
        self._rms = np.sqrt(np.mean(x ** 2, axis=1, keepdims=True) + self.eps)
        self._x_norm = x / self._rms
        return self.gamma * self._x_norm

    def backward(self, grad, pre_activation: bool = False):
        if self.autograd:
            return grad  # no-op
        D = self._input.shape[1]

        self._dgamma = np.sum(grad * self._x_norm, axis=0)

        #u = weighted gradient, dot = per-sample scalar
        u = self.gamma * grad
        dot = np.sum(u * self._x_norm, axis=1, keepdims=True) / D
        dx = (u - self._x_norm * dot) / self._rms
        return dx

    def zero_grad(self) -> None:
        if self.autograd and self.gamma is not None:
            self.gamma.zero_grad()

    def get_params(self) -> Dict[str, np.ndarray]:
        if self.gamma is None:
            return {}
        if self.autograd:
            return {"gamma": self.gamma.data}
        return {"gamma": self.gamma}

    def get_grads(self) -> Dict[str, np.ndarray]:
        if self.autograd and self.gamma is not None:
            return {"gamma": self.gamma.grad}
        if self._dgamma is None:
            return {}
        return {"gamma": self._dgamma}

    def get_weights(self) -> Dict[str, np.ndarray]:
        if self.gamma is None:
            return {}
        if self.autograd:
            return {"gamma": self.gamma.data}
        return {"gamma": self.gamma}

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        if self.autograd:
            from autograd import Tensor
            self.gamma = Tensor(weights["gamma"], requires_grad=True)
        else:
            self.gamma = weights["gamma"]

    def get_config(self) -> Dict[str, Any]:
        return {"type": "RMSNorm", "eps": self.eps}

    def __repr__(self) -> str:
        built = f"gamma{self.gamma.shape}" if self.gamma is not None else "unbuilt"
        return f"RMSNorm(eps={self.eps}, {built})"

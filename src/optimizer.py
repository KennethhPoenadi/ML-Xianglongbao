from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class Optimizer(ABC):
    def __init__(self, lr: float) -> None:
        if lr <= 0:
            raise ValueError(f"learning rate harus > 0, got {lr}")
        self.lr: float = lr

    @abstractmethod
    def step(self, layer: Any) -> None: ...

    @abstractmethod
    def get_config(self) -> Dict[str, Any]: ...

    def reset(self) -> None:
        pass


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        super().__init__(lr)

    def step(self, layer: Any) -> None:
        params = layer.get_params()
        grads = layer.get_grads()
        for name, param in params.items():
            param -= self.lr * grads[name]

    def get_config(self) -> Dict[str, Any]:
        return {"name": "sgd", "lr": self.lr}

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr})"


class Adam(Optimizer):
    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(lr)
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.eps: float = eps
        #state disimpen per layer pake id(layer) sebagai key
        self._state: Dict[int, Dict[str, Any]] = {}

    def _get_or_init_state(
        self, layer_id: int, params: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        if layer_id not in self._state:
            state: Dict[str, Any] = {"t": 0}
            for name, param in params.items():
                state[name] = {
                    "m": np.zeros_like(param),
                    "v": np.zeros_like(param),
                }
            self._state[layer_id] = state
        return self._state[layer_id]

    def step(self, layer: Any) -> None:
        params = layer.get_params()
        grads = layer.get_grads()

        state = self._get_or_init_state(id(layer), params)
        state["t"] += 1
        t = state["t"]

        b1, b2, eps = self.beta1, self.beta2, self.eps
        #bias correction
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t

        for name, param in params.items():
            g = grads[name]

            #update momen
            m = b1 * state[name]["m"] + (1.0 - b1) * g
            v = b2 * state[name]["v"] + (1.0 - b2) * (g * g)
            state[name]["m"] = m
            state[name]["v"] = v

            #bias corrected
            m_hat = m / bc1
            v_hat = v / bc2

            #update param
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def reset(self) -> None:
        self._state.clear()

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "adam",
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
        }

    def __repr__(self) -> str:
        return (
            f"Adam(lr={self.lr}, beta1={self.beta1}, "
            f"beta2={self.beta2}, eps={self.eps})"
        )


def get_optimizer(config: Dict[str, Any]) -> Optimizer:
    cfg = {k: v for k, v in config.items() if k != "name"}
    name = config["name"].lower()

    if name == "sgd":
        return SGD(**cfg)
    if name == "adam":
        return Adam(**cfg)

    raise ValueError(f"optimizer '{name}' ga dikenal. pilih 'sgd' atau 'adam'")

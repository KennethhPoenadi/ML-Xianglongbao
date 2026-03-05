from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from activation import Linear, ReLU, Sigmoid, Tanh, Softmax

# ---------------------------------------------------------------------------
# Activation registry
# ---------------------------------------------------------------------------

_ACTIVATION_MAP: Dict[str, Any] = {
    "linear":  Linear,
    "relu":    ReLU,
    "sigmoid": Sigmoid,
    "tanh":    Tanh,
    "softmax": Softmax,
}


def get_activation(name: str):
    """Return an activation instance by string name."""
    key = name.lower()
    if key not in _ACTIVATION_MAP:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Available: {list(_ACTIVATION_MAP.keys())}"
        )
    return _ACTIVATION_MAP[key]()


# ---------------------------------------------------------------------------
# Base Layer
# ---------------------------------------------------------------------------

class Layer(ABC):
    """Abstract base class for all layers."""

    @abstractmethod
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass: compute and return layer output."""

    @abstractmethod
    def backward(self, grad: np.ndarray, pre_activation: bool = False) -> np.ndarray:
        """Backward pass: return gradient w.r.t. layer input.

        Parameters
        ----------
        grad : np.ndarray
            Gradient w.r.t. the layer *output* (dL/dA) when
            ``pre_activation=False``.  When ``pre_activation=True``, grad is
            treated as dL/dZ (i.e. the activation-derivative step is skipped).
            The pre-activation path is used for the combined softmax + CCE
            gradient shortcut.
        pre_activation : bool
            If True, skip the activation derivative multiplication.
        """

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Return a dict of weight arrays (empty for parameter-free layers)."""
        return {}

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Restore weights from a dict returned by get_weights()."""

    def get_config(self) -> Dict[str, Any]:
        """Return a JSON-serialisable config dict for model persistence."""
        return {}


# ---------------------------------------------------------------------------
# Dense Layer
# ---------------------------------------------------------------------------

class Dense(Layer):
    """Fully-connected (dense) layer.

    Parameters
    ----------
    units : int
        Number of neurons / output features.
    activation : str
        Activation function name ('relu', 'sigmoid', 'tanh', 'softmax',
        'linear').
    l1 : float
        L1 regularisation coefficient (Lasso).
    l2 : float
        L2 regularisation coefficient (Ridge).
    init : str
        Weight initialisation method:
        'zeros'   — all weights set to 0.
        'uniform' — uniform random in [low, high].
        'normal'  — Gaussian with given mean and std.
        'xavier'  — Xavier/Glorot (recommended for sigmoid/tanh).
        'he'      — He initialisation (recommended for ReLU).
        Defaults to 'auto' which picks He for relu, Xavier otherwise.
    init_params : dict, optional
        Extra params depending on init method:
        uniform → {'low': float, 'high': float, 'seed': int}
        normal  → {'mean': float, 'std': float, 'seed': int}
    """

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

        # Weights — initialised in build()
        self.W: Optional[np.ndarray] = None   # shape (in_features, units)
        self.b: Optional[np.ndarray] = None   # shape (1, units)

        # Cached forward-pass values for backprop
        self._input: Optional[np.ndarray] = None
        self._z:     Optional[np.ndarray] = None  # pre-activation

        # Gradients (set during backward)
        self._dW: Optional[np.ndarray] = None
        self._db: Optional[np.ndarray] = None

        # Adam optimizer moments
        self._mW: Optional[np.ndarray] = None
        self._vW: Optional[np.ndarray] = None
        self._mb: Optional[np.ndarray] = None
        self._vb: Optional[np.ndarray] = None
        self._t:  int = 0  # Adam step counter

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, input_size: int) -> None:
        """Initialise weights given the number of input features."""
        method = self.init
        p      = self.init_params

        # Seed handling (per-layer, optional)
        seed = p.get("seed", None)
        rng  = np.random.default_rng(seed)

        if method == "zeros":
            W = np.zeros((input_size, self.units))

        elif method == "uniform":
            low  = float(p.get("low",  -0.1))
            high = float(p.get("high",  0.1))
            W    = rng.uniform(low, high, size=(input_size, self.units))

        elif method == "normal":
            mean = float(p.get("mean", 0.0))
            std  = float(p.get("std",  0.01))
            W    = rng.normal(mean, std, size=(input_size, self.units))

        elif method == "xavier":
            # Xavier / Glorot: std = sqrt(1 / fan_in)
            std = np.sqrt(1.0 / input_size)
            W   = rng.normal(0.0, std, size=(input_size, self.units))

        elif method == "he":
            # He: std = sqrt(2 / fan_in)
            std = np.sqrt(2.0 / input_size)
            W   = rng.normal(0.0, std, size=(input_size, self.units))

        else:  # "auto" — pick based on activation
            if self.activation_name == "relu":
                std = np.sqrt(2.0 / input_size)
            else:
                std = np.sqrt(1.0 / input_size)
            W = np.random.randn(input_size, self.units) * std

        self.W = W
        self.b = np.zeros((1, self.units))

        # Initialise Adam moments to zero
        self._mW = np.zeros_like(self.W)
        self._vW = np.zeros_like(self.W)
        self._mb = np.zeros_like(self.b)
        self._vb = np.zeros_like(self.b)
        self._t  = 0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Compute Z = x @ W + b  then  A = activation(Z)."""
        if self.W is None:
            self.build(x.shape[1])

        self._input = x
        self._z = x @ self.W + self.b
        return self.activation(self._z)

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward(
        self, grad: np.ndarray, pre_activation: bool = False
    ) -> np.ndarray:
        """Compute gradients for W, b and the layer input.

        Parameters
        ----------
        grad : np.ndarray
            dL/dA  (post-activation gradient) when pre_activation=False.
            dL/dZ  (pre-activation gradient)  when pre_activation=True.
        pre_activation : bool
            Skip activation derivative (used with combined softmax+CCE grad).

        Returns
        -------
        np.ndarray
            dL/dX — gradient to propagate to the previous layer.
        """
        if pre_activation:
            dz = grad
        else:
            dz = grad * self.activation.derivative(self._z)

        # Regularisation gradients
        reg_grad = self.l1 * np.sign(self.W) + self.l2 * self.W

        self._dW = self._input.T @ dz + reg_grad  # (in_features, units)
        self._db = np.sum(dz, axis=0, keepdims=True)  # (1, units)
        dx = dz @ self.W.T  # (batch, in_features)
        return dx

    # ------------------------------------------------------------------
    # Parameter updates
    # ------------------------------------------------------------------

    def update_sgd(self, lr: float) -> None:
        """Apply vanilla stochastic gradient descent."""
        self.W -= lr * self._dW
        self.b -= lr * self._db

    def update_adam(
        self,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps:   float = 1e-8,
    ) -> None:
        """Apply Adam update rule."""
        self._t += 1
        t = self._t

        # Biased first & second moment estimates
        self._mW = beta1 * self._mW + (1.0 - beta1) * self._dW
        self._vW = beta2 * self._vW + (1.0 - beta2) * (self._dW ** 2)
        self._mb = beta1 * self._mb + (1.0 - beta1) * self._db
        self._vb = beta2 * self._vb + (1.0 - beta2) * (self._db ** 2)

        # Bias-corrected estimates
        mW_hat = self._mW / (1.0 - beta1 ** t)
        vW_hat = self._vW / (1.0 - beta2 ** t)
        mb_hat = self._mb / (1.0 - beta1 ** t)
        vb_hat = self._vb / (1.0 - beta2 ** t)

        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Return current weight arrays."""
        if self.W is None:
            return {}
        return {"W": self.W, "b": self.b}

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Restore weights and reset Adam moments."""
        self.W = weights["W"]
        self.b = weights["b"]
        # Reset Adam moments (fresh state after loading)
        self._mW = np.zeros_like(self.W)
        self._vW = np.zeros_like(self.W)
        self._mb = np.zeros_like(self.b)
        self._vb = np.zeros_like(self.b)
        self._t  = 0

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
        built = f"W{self.W.shape}" if self.W is not None else "unbuilt"
        return (
            f"Dense(units={self.units}, activation='{self.activation_name}', "
            f"l1={self.l1}, l2={self.l2}, {built})"
        )


# ---------------------------------------------------------------------------
# RMSNorm Layer
# ---------------------------------------------------------------------------

class RMSNorm(Layer):
    """Root Mean Square Layer Normalisation (RMSNorm).

    Each sample is normalised by its RMS across features, then scaled
    element-wise by a learnable parameter *gamma* (no additive bias, unlike
    standard LayerNorm).

    Forward:
        rms(x)  = sqrt( mean(x^2) + eps )   shape (batch, 1)
        x_norm  = x / rms(x)                shape (batch, features)
        y       = gamma * x_norm            shape (batch, features)

    Backward (via chain-rule):
        dL/d_gamma = sum_batch( grad * x_norm )
        dL/dX_j    = (1/rms) * [ (gamma*grad)_j
                                  - x_norm_j * mean_f( gamma*grad * x_norm ) ]

    Parameters
    ----------
    eps : float
        Small constant added under the square root for numerical stability.
        Default 1e-8.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps: float = eps

        # Learnable scale parameter — initialised in build()
        self.gamma: Optional[np.ndarray] = None   # shape (features,)

        # Cached values for backprop
        self._input:  Optional[np.ndarray] = None   # (batch, features)
        self._rms:    Optional[np.ndarray] = None   # (batch, 1)
        self._x_norm: Optional[np.ndarray] = None   # (batch, features)

        # Gradient of gamma (set during backward)
        self._dgamma: Optional[np.ndarray] = None   # (features,)

        # Adam optimizer moments for gamma
        self._m_gamma: Optional[np.ndarray] = None
        self._v_gamma: Optional[np.ndarray] = None
        self._t: int = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, features: int) -> None:
        """Initialise gamma to ones (identity transform at start)."""
        self.gamma   = np.ones((features,))
        self._dgamma = np.zeros((features,))
        self._m_gamma = np.zeros((features,))
        self._v_gamma = np.zeros((features,))
        self._t = 0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Normalise x by its RMS and scale by gamma.

        Parameters
        ----------
        x : np.ndarray
            Input of shape (batch, features).
        training : bool
            Unused; accepted for API compatibility.

        Returns
        -------
        np.ndarray
            Normalised and scaled output, same shape as x.
        """
        if self.gamma is None:
            self.build(x.shape[1])

        self._input = x
        # rms: (batch, 1)
        self._rms = np.sqrt(np.mean(x ** 2, axis=1, keepdims=True) + self.eps)
        self._x_norm = x / self._rms          # (batch, features)
        return self.gamma * self._x_norm      # broadcast gamma (features,)

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward(
        self, grad: np.ndarray, pre_activation: bool = False
    ) -> np.ndarray:
        """Compute gradients and propagate to previous layer.

        Parameters
        ----------
        grad : np.ndarray
            dL/dY, shape (batch, features).
        pre_activation : bool
            Unused; accepted for API compatibility.

        Returns
        -------
        np.ndarray
            dL/dX, shape (batch, features).
        """
        D = self._input.shape[1]

        # dL/d_gamma — accumulated over batch
        self._dgamma = np.sum(grad * self._x_norm, axis=0)   # (features,)

        # dL/dX derivation:
        #   u   = gamma * grad               (the "weighted" gradient)
        #   dot = mean_features(u * x_norm)  (per-sample scalar)
        #   dX  = (u - x_norm * dot) / rms
        u   = self.gamma * grad                                  # (batch, D)
        dot = np.sum(u * self._x_norm, axis=1, keepdims=True) / D  # (batch,1)
        dx  = (u - self._x_norm * dot) / self._rms              # (batch, D)
        return dx

    # ------------------------------------------------------------------
    # Parameter updates
    # ------------------------------------------------------------------

    def update_sgd(self, lr: float) -> None:
        """Apply vanilla SGD to gamma."""
        self.gamma -= lr * self._dgamma

    def update_adam(
        self,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Apply Adam update rule to gamma."""
        self._t += 1
        t = self._t

        self._m_gamma = beta1 * self._m_gamma + (1.0 - beta1) * self._dgamma
        self._v_gamma = beta2 * self._v_gamma + (1.0 - beta2) * (self._dgamma ** 2)

        m_hat = self._m_gamma / (1.0 - beta1 ** t)
        v_hat = self._v_gamma / (1.0 - beta2 ** t)

        self.gamma -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def get_weights(self) -> Dict[str, np.ndarray]:
        if self.gamma is None:
            return {}
        return {"gamma": self.gamma}

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        self.gamma = weights["gamma"]
        self._m_gamma = np.zeros_like(self.gamma)
        self._v_gamma = np.zeros_like(self.gamma)
        self._t = 0

    def get_config(self) -> Dict[str, Any]:
        return {"type": "RMSNorm", "eps": self.eps}

    def __repr__(self) -> str:
        built = f"gamma{self.gamma.shape}" if self.gamma is not None else "unbuilt"
        return f"RMSNorm(eps={self.eps}, {built})"

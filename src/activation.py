import numpy as np
class Linear:
    # f(x) = x
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class ReLU:
    # f(x) = max(0, x)
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class Sigmoid:
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self(x)
        return s * (1 - s)


class Tanh:
    # Hyperbolic Tangent: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


class Softmax:
    # Softmax activation function: f(x)_i = e^(x_i) / sum(e^(x_j))
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self(x)
        return s * (1 - s)

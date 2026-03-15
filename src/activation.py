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


class LeakyReLU:
    """
    Leaky ReLU (Leaky Rectified Linear Unit)
    
    Formula:
        f(x) = x           jika x > 0
        f(x) = alpha * x   jika x <= 0
    
    """
    
    def __init__(self, alpha: float = 0.01):
        """
        Args:
            alpha: Slope untuk nilai negatif (default: 0.01)
        """
        self.alpha = alpha
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative:
            f'(x) = 1         jika x > 0
            f'(x) = alpha     jika x <= 0
        """
        return np.where(x > 0, 1, self.alpha)


class ELU:
    """
    ELU (Exponential Linear Unit)
    
    Formula:
        f(x) = x                    jika x > 0
        f(x) = alpha * (e^x - 1)    jika x <= 0
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Parameter untuk mengontrol saturasi negatif (default: 1.0)
        """
        self.alpha = alpha
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative:
            f'(x) = 1                       jika x > 0
            f'(x) = alpha * e^x             jika x <= 0
                  = f(x) + alpha            
        """
        return np.where(x > 0, 1, self.alpha * np.exp(x))

import numpy as np

_EPS = 1e-12

class Loss:
    #base class, tiap loss wajib implement forward dan backward sendiri

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def forward_autograd(self, y_pred, y_true):
        """Hitung loss menggunakan Tensor autograd (membangun computation graph)."""
        raise NotImplementedError


class MSE(Loss):
    name = "mse"

    def forward(self, y_pred, y_true):
        #(1/n) * sum((y - y_hat)^2)
        n = y_pred.shape[0]
        return float((1 / n) * np.sum((y_true - y_pred) ** 2))

    def backward(self, y_pred, y_true):
        #dl/dy_hat = (2/n)(y_hat - y)
        n = y_pred.shape[0]
        return (2 / n) * (y_pred - y_true)

    def forward_autograd(self, y_pred, y_true):
        return ((y_true - y_pred) ** 2).sum() / y_pred.shape[0]


class BinaryCrossEntropy(Loss):
    name = "binary_crossentropy"

    def forward(self, y_pred, y_true):
        #-(1/n) * sum(y*log(y_hat) + (1-y)*log(1-y_hat))
        n = y_pred.shape[0]
        y_pred = np.clip(y_pred, _EPS, 1 - _EPS)
        return float(-(1 / n) * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        ))

    def backward(self, y_pred, y_true):
        #dl/dy_hat = (1/n) * (y_hat - y) / (y_hat*(1-y_hat))
        n = y_pred.shape[0]
        y_pred = np.clip(y_pred, _EPS, 1 - _EPS)
        return (1 / n) * (y_pred - y_true) / (y_pred * (1 - y_pred))

    def forward_autograd(self, y_pred, y_true):
        p = y_pred.clip(1e-12, 1 - 1e-12)
        return -((y_true * p.log() + (1.0 - y_true) * (1.0 - p).log()).sum() / y_pred.shape[0])


class CategoricalCrossEntropy(Loss):
    name = "categorical_crossentropy"

    def forward(self, y_pred, y_true):
        #-(1/n) * sum_i sum_j (y_ij * log(y_hat_ij))
        n = y_pred.shape[0]
        y_pred = np.clip(y_pred, _EPS, 1.0)
        return float(-(1 / n) * np.sum(y_true * np.log(y_pred)))

    def backward(self, y_pred, y_true):
        #dl/dy_hat = -(1/n) * y / y_hat
        n = y_pred.shape[0]
        y_pred = np.clip(y_pred, _EPS, 1.0)
        return -(1 / n) * y_true / y_pred

    def forward_autograd(self, y_pred, y_true):
        p = y_pred.clip(1e-12, 1.0)
        return -((y_true * p.log()).sum() / y_pred.shape[0])


#registry buat bikin instance dari string nama
_REGISTRY = {cls.name: cls for cls in [MSE, BinaryCrossEntropy, CategoricalCrossEntropy]}

def get_loss(name: str) -> Loss:
    return _REGISTRY[name.lower()]()


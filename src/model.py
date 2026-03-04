"""model.py — Sequential neural-network model.

Features
--------
* add() / compile() / fit() / predict() / evaluate()  (Keras-like API)
* Verbose training log with loss, accuracy, val_loss, val_accuracy
* Optimizers: sgd, adam
* Loss functions: mse, binary_crossentropy, categorical_crossentropy
* Softmax + categorical_crossentropy combined gradient (numerically stable)
* L1 / L2 regularisation (delegated to Dense layers)
* Weight persistence:
    - save_weights(path)  / load_weights(path)   → single .npz file
    - save(dir)           / Sequential.load(dir) → weights.npz + config.json
* summary() — human-readable layer table
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from activation import Softmax
from layer import Dense, Layer
from loss import CategoricalCrossEntropy, get_loss

# ---------------------------------------------------------------------------
# Supported optimizers
# ---------------------------------------------------------------------------
_OPTIMIZERS = {"sgd", "adam"}


# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------

class Sequential:
    """A linear stack of layers.

    Example
    -------
    >>> from layer import Dense
    >>> from model import Sequential
    >>>
    >>> model = Sequential()
    >>> model.add(Dense(64, 'relu', l2=0.001))
    >>> model.add(Dense(3,  'softmax'))
    >>> model.compile('adam', 'categorical_crossentropy')
    >>> history = model.fit(X_train, y_train, epochs=50,
    ...                     validation_data=(X_val, y_val), verbose=1)
    >>> model.save('my_model')
    >>> loaded = Sequential.load('my_model')
    """

    def __init__(self) -> None:
        self.layers: List[Layer] = []
        self._loss_fn = None
        self._optimizer: Optional[str] = None
        self._compiled: bool = False
        self._built: bool = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add(self, layer: Layer) -> None:
        """Append a layer to the network."""
        if not isinstance(layer, Layer):
            raise TypeError(
                f"Expected a Layer instance, got {type(layer).__name__}"
            )
        self.layers.append(layer)

    def compile(self, optimizer: str, loss: str) -> None:
        """Configure the model for training.

        Parameters
        ----------
        optimizer : str
            'sgd' or 'adam'.
        loss : str
            'mse', 'binary_crossentropy', or 'categorical_crossentropy'.
        """
        opt = optimizer.lower()
        if opt not in _OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. Available: {_OPTIMIZERS}"
            )
        self._optimizer = opt
        self._loss_fn   = get_loss(loss)
        self._compiled  = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build(self, input_size: int) -> None:
        """Lazily build all layers (infer input sizes)."""
        size = input_size
        for layer in self.layers:
            if isinstance(layer, Dense) and layer.W is None:
                layer.build(size)
            if isinstance(layer, Dense):
                size = layer.units
        self._built = True

    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def _backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Full backward pass through all layers.

        For the common Softmax + CategoricalCrossEntropy combination, we use
        the numerically stable combined gradient dL/dZ = (y_pred - y_true)/n
        and skip the element-wise activation derivative on the last layer.
        """
        last = self.layers[-1]
        use_combined = (
            isinstance(self._loss_fn, CategoricalCrossEntropy)
            and isinstance(last, Dense)
            and isinstance(last.activation, Softmax)
        )

        if use_combined:
            # Combined softmax + CCE gradient: dL/dZ = (ŷ - y) / n
            n = y_true.shape[0]
            grad = (y_pred - y_true) / n
            grad = last.backward(grad, pre_activation=True)
            layers_to_traverse = self.layers[:-1]
        else:
            grad = self._loss_fn.backward(y_pred, y_true)
            layers_to_traverse = self.layers

        for layer in reversed(layers_to_traverse):
            grad = layer.backward(grad)

    def _update(self, lr: float) -> None:
        """Apply optimizer update to all trainable layers."""
        for layer in self.layers:
            if isinstance(layer, Dense):
                if self._optimizer == "adam":
                    layer.update_adam(lr)
                else:
                    layer.update_sgd(lr)

    def _reg_loss(self) -> float:
        """Sum of regularisation penalties across all Dense layers."""
        total = 0.0
        for layer in self.layers:
            if isinstance(layer, Dense) and layer.W is not None:
                total += layer.l1 * np.sum(np.abs(layer.W))
                total += 0.5 * layer.l2 * np.sum(layer.W ** 2)
        return total

    @staticmethod
    def _is_classification(y: np.ndarray) -> bool:
        return y.ndim == 2 and y.shape[1] > 1

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: int = 1,
        seed: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Train the model.

        Parameters
        ----------
        X : np.ndarray  shape (n_samples, n_features)
        y : np.ndarray  shape (n_samples,) or (n_samples, n_classes)
        epochs : int
        batch_size : int
        learning_rate : float
        validation_data : tuple (X_val, y_val), optional
        verbose : int
            0 = silent, 1 = one line per epoch.
        seed : int, optional
            NumPy random seed for reproducibility.

        Returns
        -------
        dict
            Training history with keys: 'loss', 'accuracy' (if classification),
            'val_loss', 'val_accuracy' (if validation_data is provided).
        """
        if not self._compiled:
            raise RuntimeError("Call model.compile() before model.fit().")
        if len(self.layers) == 0:
            raise RuntimeError("The model has no layers. Use model.add().")

        self._build(X.shape[1])

        if seed is not None:
            np.random.seed(seed)

        is_clf = self._is_classification(y)

        history: Dict[str, List[float]] = {"loss": []}
        if is_clf:
            history["accuracy"] = []
        if validation_data is not None:
            history["val_loss"] = []
            if is_clf:
                history["val_accuracy"] = []

        n = X.shape[0]

        for epoch in range(1, epochs + 1):
            # Shuffle training data each epoch
            idx  = np.random.permutation(n)
            X_sh = X[idx]
            y_sh = y[idx]

            epoch_loss  = 0.0
            n_batches   = 0

            for start in range(0, n, batch_size):
                Xb = X_sh[start : start + batch_size]
                yb = y_sh[start : start + batch_size]

                y_pred      = self._forward(Xb, training=True)
                batch_loss  = self._loss_fn.forward(y_pred, yb) + self._reg_loss()
                epoch_loss += batch_loss
                n_batches  += 1

                self._backward(y_pred, yb)
                self._update(learning_rate)

            # --- epoch-level metrics ---
            avg_loss = epoch_loss / n_batches
            history["loss"].append(float(avg_loss))

            if is_clf:
                acc = self._accuracy(X, y)
                history["accuracy"].append(acc)

            if validation_data is not None:
                Xv, yv         = validation_data
                yv_pred        = self._forward(Xv, training=False)
                val_loss       = self._loss_fn.forward(yv_pred, yv)
                history["val_loss"].append(float(val_loss))
                if is_clf:
                    val_acc = self._accuracy(Xv, yv)
                    history["val_accuracy"].append(val_acc)

            if verbose == 1:
                self._print_epoch(epoch, epochs, history, is_clf,
                                  validation_data is not None)

        return history

    def _accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self._forward(X, training=False)
        return float(np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)))

    @staticmethod
    def _print_epoch(
        epoch: int,
        epochs: int,
        history: Dict[str, List[float]],
        is_clf: bool,
        has_val: bool,
    ) -> None:
        msg = f"Epoch {epoch:>{len(str(epochs))}}/{epochs}"
        msg += f"  loss: {history['loss'][-1]:.4f}"
        if is_clf:
            msg += f"  acc: {history['accuracy'][-1]:.4f}"
        if has_val:
            msg += f"  val_loss: {history['val_loss'][-1]:.4f}"
            if is_clf and "val_accuracy" in history:
                msg += f"  val_acc: {history['val_accuracy'][-1]:.4f}"
        print(msg)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return model output for input X (inference mode, no grad)."""
        if not self._built:
            raise RuntimeError(
                "Model is not built yet. Call fit() or load_weights() first."
            )
        return self._forward(X, training=False)

    def evaluate(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Compute loss (and accuracy if classification) on a dataset.

        Returns
        -------
        dict with 'loss' and optionally 'accuracy'.
        """
        y_pred  = self.predict(X)
        results = {"loss": float(self._loss_fn.forward(y_pred, y))}
        if self._is_classification(y):
            results["accuracy"] = float(
                np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
            )
        return results

    # ------------------------------------------------------------------
    # Weight persistence
    # ------------------------------------------------------------------

    def save_weights(self, filepath: Union[str, Path]) -> None:
        """Save all trainable weights to a single .npz file.

        Parameters
        ----------
        filepath : str or Path
            Destination path. The '.npz' extension is appended if absent.

        Example
        -------
        >>> model.save_weights('checkpoints/epoch_10')
        # creates  checkpoints/epoch_10.npz
        """
        filepath = str(filepath)
        if not filepath.endswith(".npz"):
            filepath += ".npz"

        arrays: Dict[str, np.ndarray] = {}
        for i, layer in enumerate(self.layers):
            for key, val in layer.get_weights().items():
                arrays[f"layer_{i}_{key}"] = val

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        np.savez(filepath, **arrays)

    def load_weights(self, filepath: Union[str, Path]) -> None:
        """Load weights from a .npz file into the current model.

        The model architecture must match the saved weights.

        Parameters
        ----------
        filepath : str or Path
            Path to the .npz file (extension optional).
        """
        filepath = str(filepath)
        if not filepath.endswith(".npz"):
            filepath += ".npz"

        data = np.load(filepath)
        for i, layer in enumerate(self.layers):
            template = layer.get_weights()
            if not template:
                continue
            weights = {key: data[f"layer_{i}_{key}"] for key in template}
            layer.set_weights(weights)
        self._built = True

    def save(self, dirpath: Union[str, Path]) -> None:
        """Save model architecture + weights to a directory.

        Creates two files:
        - ``<dirpath>/config.json``  — layer architecture & optimizer/loss
        - ``<dirpath>/weights.npz``  — numpy weight arrays

        Parameters
        ----------
        dirpath : str or Path
            Directory to save into (created if it does not exist).

        Example
        -------
        >>> model.save('saved_models/my_model')
        >>> loaded = Sequential.load('saved_models/my_model')
        """
        if not self._compiled:
            raise RuntimeError("Compile the model before saving.")

        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        config = {
            "optimizer": self._optimizer,
            "loss":      self._loss_fn.name,
            "layers":    [layer.get_config() for layer in self.layers],
        }
        with open(dirpath / "config.json", "w") as fh:
            json.dump(config, fh, indent=2)

        self.save_weights(dirpath / "weights.npz")

    @classmethod
    def load(cls, dirpath: Union[str, Path]) -> "Sequential":
        """Load a previously saved model from a directory.

        Parameters
        ----------
        dirpath : str or Path
            Directory containing ``config.json`` and ``weights.npz``.

        Returns
        -------
        Sequential
            Fully restored model ready for inference or continued training.
        """
        dirpath = Path(dirpath)

        config_path = dirpath / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"config.json not found in '{dirpath}'. "
                "Was the model saved with model.save()?"
            )

        with open(config_path) as fh:
            config = json.load(fh)

        model = cls()
        for lc in config["layers"]:
            if lc["type"] == "Dense":
                model.add(
                    Dense(
                        units=lc["units"],
                        activation=lc.get("activation", "linear"),
                        l1=lc.get("l1", 0.0),
                        l2=lc.get("l2", 0.0),
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported layer type in config: '{lc['type']}'"
                )

        model.compile(config["optimizer"], config["loss"])
        model.load_weights(dirpath / "weights.npz")
        return model

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a summary table of the model architecture."""
        sep = "=" * 56
        print(sep)
        print(f"{'Layer (type)':<22} {'Output Shape':<18} {'Params':>10}")
        print(sep)
        total_params = 0
        for i, layer in enumerate(self.layers):
            ltype = type(layer).__name__
            name  = f"{ltype}_{i}"
            if isinstance(layer, Dense):
                out_shape = f"(None, {layer.units})"
                params    = (layer.W.size + layer.b.size) if layer.W is not None else "?"
                total_params += params if isinstance(params, int) else 0
            else:
                out_shape = "?"
                params    = 0
            print(f"{name:<22} {out_shape:<18} {str(params):>10}")
        print(sep)
        print(f"Total params: {total_params:,}")
        print(sep)

    def __repr__(self) -> str:
        compiled = (
            f"optimizer='{self._optimizer}', loss='{self._loss_fn.name}'"
            if self._compiled
            else "not compiled"
        )
        return f"Sequential(layers={len(self.layers)}, {compiled})"

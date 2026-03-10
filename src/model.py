from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from layer import Dense, Layer, RMSNorm
from loss import Loss, get_loss
from optimizer import Optimizer, SGD, get_optimizer


class Model:
    """feedforward Neural Network (FFNN) model."""
    def __init__(self) -> None:
        self.layers: List[Layer] = []
        self.loss: Optional[Loss] = None
        self.loss_name: str = ""
        self.optimizer: Optimizer = SGD(lr=0.01)
        self.autograd: bool = False

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def compile(self,loss: str,learning_rate: float = 0.01,optimizer: str = "sgd",autograd: bool = False, **optimizer_kwargs,) -> None:
        self.loss_name = loss.lower()
        self.loss = get_loss(self.loss_name)
        self.autograd = autograd
        if autograd:
            for layer in self.layers:
                layer.autograd = True
        self.optimizer = get_optimizer(
            {"name": optimizer, "lr": learning_rate, **optimizer_kwargs}
        )

    def forward(self, X, training: bool = True):
        if self.autograd:
            from autograd import Tensor
            out = X if isinstance(X, Tensor) else Tensor(X)
            for layer in self.layers:
                out = layer.forward(out, training=training)
            return out
        out = X
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def _zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        grad = self.loss.backward(y_pred, y_true)
        last_dense = next(
            (l for l in reversed(self.layers) if isinstance(l, Dense)), None
        )
        if (
            last_dense is not None
            and last_dense is self.layers[-1]
            and last_dense.activation_name == "softmax"
            and self.loss_name == "categorical_crossentropy"
        ):
            grad = (y_pred - y_true) / y_pred.shape[0]
            grad = self.layers[-1].backward(grad, pre_activation=True)
            for layer in reversed(self.layers[:-1]):
                grad = layer.backward(grad, pre_activation=False)
        else:
            for layer in reversed(self.layers):
                grad = layer.backward(grad, pre_activation=False)

    def update_weights(self) -> None:
        for layer in self.layers:
            if layer.get_params():
                self.optimizer.step(layer)

    def fit(self,X_train: np.ndarray,y_train: np.ndarray,epochs: int = 10,batch_size: int = 32,validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,verbose: int = 1,shuffle: bool = True,) -> Dict[str, List[float]]:
        if self.loss is None:
            raise RuntimeError(
                "model must be compiled before training. call model.compile()."
            )

        self.history = {"train_loss": [], "val_loss": []}

        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        for epoch in range(epochs):
            #shuffle data
            if shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]
            else:
                X_shuffled = X_train
                y_shuffled = y_train

            #training loop
            epoch_losses = []

            if verbose == 1:
                pbar = tqdm(range(n_batches),desc=f"Epoch {epoch + 1}/{epochs}",ncols=100,)
            else:
                pbar = range(n_batches)

            for batch_idx in pbar:
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                if self.autograd:
                    from autograd import Tensor
                    # 1.zero gradients
                    self._zero_grad()
                    # 2.forward
                    y_pred = self.forward(X_batch, training=True)
                    # 3.compute loss via Tensor ops (graph)
                    y_true_t = Tensor(
                        y_batch if y_batch.ndim == 2
                        else y_batch.reshape(-1, 1)
                    )
                    loss_tensor = self.loss.forward_autograd(
                        y_pred, y_true_t
                    )
                    batch_loss = float(loss_tensor.data)
                    # 4.backward
                    loss_tensor.backward()
                    # 5.update weights
                    self.update_weights()
                else:
                    #forward pass
                    y_pred = self.forward(X_batch, training=True)
                    #compute loss
                    batch_loss = self.loss.forward(y_pred, y_batch)
                    #backward pass
                    self.backward(y_pred, y_batch)
                    #update weights
                    self.update_weights()

                epoch_losses.append(batch_loss)

                if verbose == 1 and isinstance(pbar, tqdm):
                    pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

            train_loss = float(np.mean(epoch_losses))
            self.history["train_loss"].append(train_loss)

            if validation_data is not None:
                X_val, y_val = validation_data
                y_val_pred = self.predict(X_val)
                val_loss = self.loss.forward(y_val_pred, y_val)
                self.history["val_loss"].append(val_loss)

                if verbose == 1:
                    print(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"loss: {train_loss:.4f} - val_loss: {val_loss:.4f}"
                    )
            else:
                self.history["val_loss"].append(None)
                if verbose == 1:
                    print(f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f}")

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self.forward(X, training=False)
        if self.autograd:
            return out.data  # tensor → numpy
        return out

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, Optional[float]]:
        if self.loss is None:
            raise RuntimeError("Model must be compiled before evaluation.")

        y_pred = self.predict(X)
        loss = self.loss.forward(y_pred, y)

        accuracy = None
        if self.loss_name in ["binary_crossentropy", "categorical_crossentropy"]:
            if y.ndim == 2 and y.shape[1] > 1:
                #multi-class classification
                y_pred_class = np.argmax(y_pred, axis=1)
                y_true_class = np.argmax(y, axis=1)
            else:
                #binary classification
                y_pred_class = (y_pred > 0.5).astype(int).flatten()
                y_true_class = y.flatten()

            accuracy = float(np.mean(y_pred_class == y_true_class))

        return loss, accuracy

    def plot_weight_distributions(self, layer_indices: Optional[List[int]] = None) -> None:
        if layer_indices is None:
            layer_indices = list(range(len(self.layers)))

        n_plots = len(layer_indices)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        for i, idx in enumerate(layer_indices):
            if idx >= len(self.layers):
                print(f"Warning: Layer index {idx} out of range, skipping.")
                continue

            layer = self.layers[idx]
            params = layer.get_params()
            if isinstance(layer, Dense) and "W" in params:
                weights = params["W"].flatten()
                axes[i].hist(weights, bins=50, alpha=0.7, edgecolor="black")
                axes[i].set_title(f"Layer {idx} (Dense) Weight Distribution")
                axes[i].set_xlabel("Weight Value")
                axes[i].set_ylabel("Frequency")
                axes[i].grid(True, alpha=0.3)
            elif isinstance(layer, RMSNorm) and "gamma" in params:
                axes[i].hist(
                    params["gamma"], bins=30, alpha=0.7,
                    edgecolor="black", color="orange"
                )
                axes[i].set_title(f"Layer {idx} (RMSNorm) Gamma Distribution")
                axes[i].set_xlabel("Gamma Value")
                axes[i].set_ylabel("Frequency")
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(
                    0.5, 0.5, "No weights available",
                    ha="center", va="center",
                    transform=axes[i].transAxes,
                )
                axes[i].set_title(f"Layer {idx}")

        plt.tight_layout()
        plt.show()

    def plot_gradient_distributions(self, layer_indices: Optional[List[int]] = None) -> None:
        if layer_indices is None:
            layer_indices = list(range(len(self.layers)))

        n_plots = len(layer_indices)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        for i, idx in enumerate(layer_indices):
            if idx >= len(self.layers):
                print(f"Warning: Layer index {idx} out of range, skipping.")
                continue

            layer = self.layers[idx]
            grads = layer.get_grads()
            if isinstance(layer, Dense) and "W" in grads:
                gradients = grads["W"].flatten()
                axes[i].hist(gradients, bins=50, alpha=0.7, edgecolor="black")
                axes[i].set_title(f"Layer {idx} (Dense) Gradient Distribution")
                axes[i].set_xlabel("Gradient Value")
                axes[i].set_ylabel("Frequency")
                axes[i].grid(True, alpha=0.3)
            elif isinstance(layer, RMSNorm) and "gamma" in grads:
                axes[i].hist(
                    grads["gamma"], bins=30, alpha=0.7,
                    edgecolor="black", color="orange"
                )
                axes[i].set_title(
                    f"Layer {idx} (RMSNorm) dGamma Distribution"
                )
                axes[i].set_xlabel("Gradient Value")
                axes[i].set_ylabel("Frequency")
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(
                    0.5,
                    0.5,
                    "No gradients\navailable",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                axes[i].set_title(f"Layer {idx}")

        plt.tight_layout()
        plt.show()

    def save(self, filepath: str) -> None:
        """Save the model architecture and weights to a file.

        Parameters
        ----------
        filepath : str
            Path to save the model (should end with .json).
        """
        model_data = {
            "architecture": [layer.get_config() for layer in self.layers],
            "weights": [layer.get_weights() for layer in self.layers],
            "loss": self.loss_name,
            "optimizer": self.optimizer.get_config(),
        }

        for layer_weights in model_data["weights"]:
            for key in layer_weights:
                layer_weights[key] = layer_weights[key].tolist()

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        with open(filepath, "r") as f:
            model_data = json.load(f)
        self.layers = []
        for layer_config in model_data["architecture"]:
            if layer_config["type"] == "Dense":
                layer = Dense(
                    units=layer_config["units"],
                    activation=layer_config["activation"],
                    l1=layer_config.get("l1", 0.0),
                    l2=layer_config.get("l2", 0.0),
                    init=layer_config.get("init", "auto"),
                    init_params=layer_config.get("init_params", {}),
                )
                self.layers.append(layer)
            elif layer_config["type"] == "RMSNorm":
                layer = RMSNorm(eps=layer_config.get("eps", 1e-8))
                self.layers.append(layer)

        for layer, weights_dict in zip(self.layers, model_data["weights"]):
            if weights_dict:
                weights_np = {
                    k: np.array(v) for k, v in weights_dict.items()
                }
                layer.set_weights(weights_np)

        self.loss_name = model_data["loss"]
        self.loss = get_loss(self.loss_name)
        self.optimizer = get_optimizer(model_data.get(
            "optimizer", {"name": "sgd", "lr": model_data.get("learning_rate", 0.01)}
        ))
        self.optimizer.reset()

        print(f"Model loaded from {filepath}")

    def summary(self, input_shape: Optional[int] = None) -> None:
        print("=" * 70)
        print("Model Summary")
        print("=" * 70)
        print(f"{'Layer':<20} {'Output Shape':<20} {'Param #':<15}")
        print("-" * 70)

        total_params = 0
        prev_size = input_shape
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                output_shape = f"(None, {layer.units})"
                params = layer.get_params()
                if params:
                    num_params = params["W"].size + params["b"].size
                elif prev_size is not None:
                    num_params = prev_size * layer.units + layer.units
                else:
                    num_params = 0
                prev_size = layer.units

                print(
                    f"Dense_{i:<14} {output_shape:<20} {num_params:<15,}"
                )
                total_params += num_params

            elif isinstance(layer, RMSNorm):
                params = layer.get_params()
                if params:
                    output_shape = f"(None, {params['gamma'].shape[0]})"
                    num_params = params['gamma'].size
                else:
                    output_shape = "(None, ?)"
                    num_params = 0
                print(
                    f"RMSNorm_{i:<13} {output_shape:<20} {num_params:<15,}"
                )
                total_params += num_params

        print("=" * 70)
        print(f"Total params: {total_params:,}")
        print("=" * 70)

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return f"Model(layers={len(self.layers)}, loss={self.loss_name})"
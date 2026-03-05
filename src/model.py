from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from layer import Dense, Layer, RMSNorm
from loss import Loss, get_loss


class Model:
    """Feedforward Neural Network (FFNN) model."""

    def __init__(self) -> None:
        """Initialize an empty model."""
        self.layers: List[Layer] = []
        self.loss: Optional[Loss] = None
        self.loss_name: str = ""
        self.learning_rate: float = 0.01

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    def add(self, layer: Layer) -> None:
        """Add a layer to the model.

        Parameters
        ----------
        layer : Layer
            The layer to add (e.g., Dense layer).
        """
        self.layers.append(layer)

    def compile(
        self,
        loss: str,
        learning_rate: float = 0.01,
    ) -> None:
        """Compile the model with a loss function and learning rate.

        Parameters
        ----------
        loss : str
            Loss function name: 'mse', 'binary_crossentropy',
            'categorical_crossentropy'.
        learning_rate : float
            Learning rate for gradient descent. Default is 0.01.
        """
        self.loss_name = loss.lower()
        self.loss = get_loss(self.loss_name)
        self.learning_rate = learning_rate

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Perform forward propagation through all layers.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (batch_size, input_features).
        training : bool
            Whether the model is in training mode. Default is True.

        Returns
        -------
        np.ndarray
            Model output of shape (batch_size, output_features).
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Perform backward propagation through all layers.

        Computes gradients for all weights using the chain rule.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted output from forward pass, shape (batch_size, num_classes).
        y_true : np.ndarray
            True labels, shape (batch_size, num_classes).
        """
        # Gradient of loss w.r.t. output
        grad = self.loss.backward(y_pred, y_true)

        # Special case: Softmax + Categorical Cross-Entropy shortcut
        # The combined derivative simplifies to (y_pred - y_true)
        # Find last Dense layer (skip trailing normalisation layers)
        last_dense = next(
            (l for l in reversed(self.layers) if isinstance(l, Dense)), None
        )
        if (
            last_dense is not None
            and last_dense is self.layers[-1]
            and last_dense.activation_name == "softmax"
            and self.loss_name == "categorical_crossentropy"
        ):
            # Use the simplified gradient: dL/dz = y_pred - y_true
            grad = (y_pred - y_true) / y_pred.shape[0]
            # Backprop through last layer with pre_activation=True
            grad = self.layers[-1].backward(grad, pre_activation=True)
            # Backprop through remaining layers normally
            for layer in reversed(self.layers[:-1]):
                grad = layer.backward(grad, pre_activation=False)
        else:
            # Standard backprop through all layers
            for layer in reversed(self.layers):
                grad = layer.backward(grad, pre_activation=False)

    def update_weights(self) -> None:
        """Update weights in all layers using gradient descent."""
        for layer in self.layers:
            if isinstance(layer, (Dense, RMSNorm)):
                layer.update_sgd(self.learning_rate)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: int = 1,
        shuffle: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the model on the given dataset.

        Parameters
        ----------
        X_train : np.ndarray
            Training input data, shape (num_samples, num_features).
        y_train : np.ndarray
            Training labels, shape (num_samples, num_classes) or (num_samples,).
        epochs : int
            Number of training epochs. Default is 10.
        batch_size : int
            Number of samples per gradient update. Default is 32.
        validation_data : tuple of (X_val, y_val), optional
            Validation data to evaluate loss at the end of each epoch.
        verbose : int
            Verbosity mode:
            0 = silent
            1 = progress bar with train/val loss
        shuffle : bool
            Whether to shuffle training data before each epoch. Default True.

        Returns
        -------
        dict
            History containing 'train_loss' and 'val_loss' per epoch.
        """
        if self.loss is None:
            raise RuntimeError(
                "Model must be compiled before training. Call model.compile()."
            )

        # Reset history
        self.history = {"train_loss": [], "val_loss": []}

        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        for epoch in range(epochs):
            # Shuffle data
            if shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]
            else:
                X_shuffled = X_train
                y_shuffled = y_train

            # Training loop
            epoch_losses = []

            if verbose == 1:
                pbar = tqdm(
                    range(n_batches),
                    desc=f"Epoch {epoch + 1}/{epochs}",
                    ncols=100,
                )
            else:
                pbar = range(n_batches)

            for batch_idx in pbar:
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass
                y_pred = self.forward(X_batch, training=True)

                # Compute loss
                batch_loss = self.loss.forward(y_pred, y_batch)
                epoch_losses.append(batch_loss)

                # Backward pass
                self.backward(y_pred, y_batch)

                # Update weights
                self.update_weights()

                # Update progress bar
                if verbose == 1 and isinstance(pbar, tqdm):
                    pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

            # Compute epoch metrics
            train_loss = float(np.mean(epoch_losses))
            self.history["train_loss"].append(train_loss)

            # Validation loss
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
        """Generate predictions for input samples.

        Parameters
        ----------
        X : np.ndarray
            Input data, shape (num_samples, num_features).

        Returns
        -------
        np.ndarray
            Predictions, shape (num_samples, num_classes).
        """
        return self.forward(X, training=False)

    def evaluate(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[float, Optional[float]]:
        """Evaluate the model on test data.

        Parameters
        ----------
        X : np.ndarray
            Test input data.
        y : np.ndarray
            Test labels.

        Returns
        -------
        tuple of (loss, accuracy)
            Test loss and accuracy (if applicable).
        """
        if self.loss is None:
            raise RuntimeError("Model must be compiled before evaluation.")

        y_pred = self.predict(X)
        loss = self.loss.forward(y_pred, y)

        # Compute accuracy for classification tasks
        accuracy = None
        if self.loss_name in ["binary_crossentropy", "categorical_crossentropy"]:
            if y.ndim == 2 and y.shape[1] > 1:
                # Multi-class classification
                y_pred_class = np.argmax(y_pred, axis=1)
                y_true_class = np.argmax(y, axis=1)
            else:
                # Binary classification
                y_pred_class = (y_pred > 0.5).astype(int).flatten()
                y_true_class = y.flatten()

            accuracy = float(np.mean(y_pred_class == y_true_class))

        return loss, accuracy

    def plot_weight_distributions(
        self, layer_indices: Optional[List[int]] = None
    ) -> None:
        """Plot weight distributions for specified layers.

        Parameters
        ----------
        layer_indices : list of int, optional
            Indices of layers to plot (0-indexed). If None, plots all layers.
        """
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
            if isinstance(layer, Dense) and layer.W is not None:
                weights = layer.W.flatten()
                label = f"Dense_{idx} W"
                axes[i].hist(weights, bins=50, alpha=0.7, edgecolor="black")
                axes[i].set_title(f"Layer {idx} (Dense) Weight Distribution")
                axes[i].set_xlabel("Weight Value")
                axes[i].set_ylabel("Frequency")
                axes[i].grid(True, alpha=0.3)
            elif isinstance(layer, RMSNorm) and layer.gamma is not None:
                axes[i].hist(
                    layer.gamma, bins=30, alpha=0.7,
                    edgecolor="black", color="orange"
                )
                axes[i].set_title(f"Layer {idx} (RMSNorm) Gamma Distribution")
                axes[i].set_xlabel("Gamma Value")
                axes[i].set_ylabel("Frequency")
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(
                    0.5, 0.5, "No weights\navailable",
                    ha="center", va="center",
                    transform=axes[i].transAxes,
                )
                axes[i].set_title(f"Layer {idx}")

        plt.tight_layout()
        plt.show()

    def plot_gradient_distributions(
        self, layer_indices: Optional[List[int]] = None
    ) -> None:
        """Plot gradient distributions for specified layers.

        Parameters
        ----------
        layer_indices : list of int, optional
            Indices of layers to plot (0-indexed). If None, plots all layers.

        Notes
        -----
        Gradients must be computed via backward pass before calling this method.
        """
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
            if isinstance(layer, Dense) and layer._dW is not None:
                gradients = layer._dW.flatten()
                axes[i].hist(gradients, bins=50, alpha=0.7, edgecolor="black")
                axes[i].set_title(f"Layer {idx} (Dense) Gradient Distribution")
                axes[i].set_xlabel("Gradient Value")
                axes[i].set_ylabel("Frequency")
                axes[i].grid(True, alpha=0.3)
            elif isinstance(layer, RMSNorm) and layer._dgamma is not None:
                axes[i].hist(
                    layer._dgamma, bins=30, alpha=0.7,
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
            "learning_rate": self.learning_rate,
        }

        # Convert numpy arrays to lists for JSON serialization
        for layer_weights in model_data["weights"]:
            for key in layer_weights:
                layer_weights[key] = layer_weights[key].tolist()

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model architecture and weights from a file.

        Parameters
        ----------
        filepath : str
            Path to the saved model file.
        """
        with open(filepath, "r") as f:
            model_data = json.load(f)

        # Clear existing layers
        self.layers = []

        # Reconstruct layers
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

        # Restore weights
        for layer, weights_dict in zip(self.layers, model_data["weights"]):
            if weights_dict:
                # Convert lists back to numpy arrays
                weights_np = {
                    k: np.array(v) for k, v in weights_dict.items()
                }
                layer.set_weights(weights_np)

        # Restore compilation settings
        self.compile(
            loss=model_data["loss"],
            learning_rate=model_data["learning_rate"],
        )

        print(f"Model loaded from {filepath}")

    def summary(self) -> None:
        """Print a summary of the model architecture."""
        print("=" * 70)
        print("Model Summary")
        print("=" * 70)
        print(f"{'Layer':<20} {'Output Shape':<20} {'Param #':<15}")
        print("-" * 70)

        total_params = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                if layer.W is not None:
                    output_shape = f"(None, {layer.units})"
                    num_params = layer.W.size + layer.b.size
                else:
                    output_shape = f"(None, {layer.units})"
                    num_params = 0

                print(
                    f"Dense_{i:<14} {output_shape:<20} {num_params:<15,}"
                )
                total_params += num_params

            elif isinstance(layer, RMSNorm):
                if layer.gamma is not None:
                    output_shape = f"(None, {layer.gamma.shape[0]})"
                    num_params = layer.gamma.size
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

# USAGE EXAMPLE
if __name__ == "__main__":
    # Create a simple model
    model = Model()
    model.add(Dense(units=64, activation="relu", init="he"))
    model.add(Dense(units=10, activation="softmax", init="xavier"))
    model.compile(loss="categorical_crossentropy", learning_rate=0.01)

    dummy_input = np.zeros((1, 784))
    _ = model.forward(dummy_input)

    # Print model summary
    model.summary()
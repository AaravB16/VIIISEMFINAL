"""Deep learning models for CAN bus intrusion detection.

Provides MLP, LSTM, and 1D-CNN architectures built with PyTorch.
A thin sklearn-compatible wrapper (``KerasClassifierWrapper``) is
provided so that the models integrate seamlessly with the existing
train / predict / evaluate pipeline, keeping the same API name
to avoid refactoring other modules.
"""

import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch not installed — deep learning models will be unavailable. "
        "Install with: pip install torch"
    )


def _check_tf():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for deep learning models. "
            "Install with: pip install torch"
        )


# -----------------------------------------------------------------------
# PyTorch Architectures
# -----------------------------------------------------------------------

class PyTorchMLP(nn.Module):
    """Multi-Layer Perceptron classifier in PyTorch."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        layers = []
        prev_dim = input_dim
        for units in hidden_layers:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = units
        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PyTorchLSTM(nn.Module):
    """LSTM sequence classifier in PyTorch."""

    def __init__(
        self,
        window_size: int,
        n_features: int,
        n_classes: int,
        lstm_units: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if lstm_units is None:
            lstm_units = [64, 32]

        self.lstm_layers = nn.ModuleList()
        prev_dim = n_features
        for units in lstm_units:
            self.lstm_layers.append(
                nn.LSTM(prev_dim, units, batch_first=True)
            )
            prev_dim = units

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, window_size, n_features)
        out = x
        for lstm in self.lstm_layers:
            out, _ = lstm(out)
        # Take output of the last sequence step
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


class PyTorchCNN1D(nn.Module):
    """1D-CNN classifier in PyTorch."""

    def __init__(
        self,
        window_size: int,
        n_features: int,
        n_classes: int,
        filters: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        if filters is None:
            filters = [64, 128]

        self.conv_layers = nn.ModuleList()
        prev_channels = n_features
        for f in filters:
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(prev_channels, f, kernel_size, padding='same'),
                nn.BatchNorm1d(f),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, padding=1),
                nn.Dropout(dropout)
            ))
            prev_channels = f

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(prev_channels, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, window_size, n_features)
        # Conv1d expects (batch, channels, length) -> transpose to (batch, n_features, window_size)
        x = x.transpose(1, 2)
        out = x
        for conv in self.conv_layers:
            out = conv(out)
        out = self.global_pool(out).squeeze(-1)
        return self.fc(out)


# -----------------------------------------------------------------------
# Sklearn-compatible wrapper (maintaining KerasClassifierWrapper name)
# -----------------------------------------------------------------------

class KerasClassifierWrapper:
    """sklearn-compatible wrapper that runs PyTorch models under the hood.

    Keeps the class name `KerasClassifierWrapper` to prevent refactoring
    in other pipeline files.
    """

    def __init__(
        self,
        build_fn: str,
        build_kwargs: dict,
        epochs: int = 20,
        batch_size: int = 256,
        validation_split: float = 0.1,
        seed: int = 42,
    ):
        self.build_fn_name = build_fn
        self.build_kwargs = build_kwargs
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.seed = seed
        self.model_ = None
        self.classes_ = None
        self.is_sequential = build_fn in ("lstm", "cnn1d")

    def fit(self, X: np.ndarray, y: np.ndarray):
        _check_tf()
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Build PyTorch model based on type
        if self.build_fn_name == "mlp":
            self.model_ = PyTorchMLP(
                input_dim=X.shape[1],
                n_classes=n_classes,
                **self.build_kwargs,
            )
        elif self.build_fn_name == "lstm":
            window_size, n_features = X.shape[1], X.shape[2]
            self.model_ = PyTorchLSTM(
                window_size=window_size,
                n_features=n_features,
                n_classes=n_classes,
                **self.build_kwargs,
            )
        elif self.build_fn_name == "cnn1d":
            window_size, n_features = X.shape[1], X.shape[2]
            self.model_ = PyTorchCNN1D(
                window_size=window_size,
                n_features=n_features,
                n_classes=n_classes,
                **self.build_kwargs,
            )
        else:
            raise ValueError(f"Unknown deep learning model: {self.build_fn_name}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_.to(device)

        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters())

        self.model_.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = self.model_(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        _check_tf()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = self.model_(X_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _check_tf()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = self.model_(X_tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba


def get_deep_learning_model(
    model_name: str,
    params: dict,
    seed: int = 42,
) -> KerasClassifierWrapper:
    """Factory function for deep learning models."""
    _check_tf()

    # Separate training params from architecture params
    training_keys = {"epochs", "batch_size", "validation_split"}
    training_params = {k: params[k] for k in training_keys if k in params}
    build_kwargs = {k: v for k, v in params.items() if k not in training_keys}

    return KerasClassifierWrapper(
        build_fn=model_name,
        build_kwargs=build_kwargs,
        seed=seed,
        **training_params,
    )


# Deep learning model names for identification
DL_MODEL_NAMES = {"mlp", "lstm", "cnn1d"}


def is_deep_learning_model(model_name: str) -> bool:
    """Check whether a model name refers to a deep learning model."""
    return model_name in DL_MODEL_NAMES

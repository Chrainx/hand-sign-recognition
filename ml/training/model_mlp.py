import torch
import torch.nn as nn


class DigitMLP(nn.Module):
    """
    Multi-Layer Perceptron for 10-class hand sign classification.

    Input:
        63-dimensional wrist-normalized landmark vector

    Output:
        Raw logits of shape (batch_size, num_classes)

    Designed to be lightweight for real-time inference.
    """

    def __init__(
        self,
        input_dim: int = 63,
        hidden_dims: tuple[int, ...] = (128, 64),
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("Input must be 2D tensor (batch_size, feature_dim)")
        return self.network(x)
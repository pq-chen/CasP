from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module


class SinusoidalPositionalEncoding(Module):
    def __init__(
        self, dim: int, train_size: int, test_size: Optional[int] = None
    ) -> None:
        super().__init__()
        self.train_size = train_size
        self.test_size = test_size

        theta = 10000
        factor = (
            -torch.tensor(theta).log() * torch.arange(0, dim, 4) / dim
        ).exp()

        max_shape = 256, 256
        freqs_y = torch.ones(*max_shape, 1).cumsum(0) * factor
        freqs_x = torch.ones(*max_shape, 1).cumsum(1) * factor
        if test_size is not None and test_size > train_size:
            freqs_y = freqs_y * (train_size - 1) / (test_size - 1)
            freqs_x = freqs_x * (train_size - 1) / (test_size - 1)
        if test_size is None:
            self.register_buffer("freqs_y", freqs_y, persistent=False)
            self.register_buffer("freqs_x", freqs_x, persistent=False)

        freqs_sin = torch.stack(
            [t.sin() for t in [freqs_y, freqs_x]], dim=-1
        ).flatten(start_dim=-2)
        freqs_cos = torch.stack(
            [t.cos() for t in [freqs_y, freqs_x]], dim=-1
        ).flatten(start_dim=-2)
        encoding = torch.stack([freqs_sin, freqs_cos], dim=-1).flatten(
            start_dim=-2
        )
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        test_size = max(x.shape[-2:])
        if self.test_size is None and test_size > self.train_size:
            freqs_y = self.freqs_y * (self.train_size - 1) / (test_size - 1)
            freqs_x = self.freqs_x * (self.train_size - 1) / (test_size - 1)
            freqs_sin = torch.stack(
                [t.sin() for t in [freqs_y, freqs_x]], dim=-1
            ).flatten(start_dim=-2)
            freqs_cos = torch.stack(
                [t.cos() for t in [freqs_y, freqs_x]], dim=-1
            ).flatten(start_dim=-2)
            encoding = torch.stack([freqs_sin, freqs_cos], dim=-1).flatten(
                start_dim=-2
            )
            return encoding
        else:
            return self.encoding

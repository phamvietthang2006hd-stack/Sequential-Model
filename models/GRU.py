import torch
from torch import Tensor, nn

from .base import RecurrentCell


class GRUCell(RecurrentCell):
    """GRU cell using the update-gate convention of ``torch.nn.GRUCell``."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_layer = nn.Linear(input_dim, hidden_dim * 3)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim * 3)

    def forward(self, x_t: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute one GRU transition.

        Args:
            x_t: Input vector, shape ``(batch_size, input_dim)``.
            state: Previous hidden state, shape ``(batch_size, hidden_dim)``.
        """

        h_prev = state

        x = self.input_layer(x_t)
        h = self.hidden_layer(h_prev)

        xz, xr, xn = x.chunk(3, dim=-1)
        hz, hr, hn = h.chunk(3, dim=-1)

        z_t = torch.sigmoid(xz + hz)
        r_t = torch.sigmoid(xr + hr)

        n_t = torch.tanh(xn + r_t * hn)

        h_t = (1 - z_t) * n_t + z_t * h_prev

        return h_t, h_t

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

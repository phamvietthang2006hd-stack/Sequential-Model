import torch
from torch import Tensor, nn

from .base import RecurrentCell


class LSTMCell(RecurrentCell):
    """LSTM cell with input, forget, candidate, and output gates."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim * 4)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim * 4)

    def forward(
        self, x_t: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Compute one LSTM transition from ``(h_{t-1}, c_{t-1})``."""
        h_prev, c_prev = state
        x_i, x_f, x_g, x_o = self.input_layer(x_t).chunk(4, dim=-1)
        h_i, h_f, h_g, h_o = self.hidden_layer(h_prev).chunk(4, dim=-1)

        input_gate = torch.sigmoid(x_i + h_i)
        forget_gate = torch.sigmoid(x_f + h_f)
        candidate = torch.tanh(x_g + h_g)
        output_gate = torch.sigmoid(x_o + h_o)

        c_t = forget_gate * c_prev + input_gate * candidate
        h_t = output_gate * torch.tanh(c_t)
        return h_t, (h_t, c_t)

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        state = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        return state, state.clone()

import torch
from torch import nn, Tensor

from .base import RecurrentCell

class RNNCell(RecurrentCell):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_t: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute one RNN transition
        @params:
            x_t: input vector, shape (batch_size, input_dim)
            state: previous state, shape (batch_size, hidden_dim)

        return:
            h_t: Current state, shape: (batch_size, hidden_dim)
            new_state: shape (batch_size, hidden_dim)
        """
        h_t = torch.tanh(self.input_layer(x_t) + self.hidden_layer(state))
        new_state = h_t
        return h_t, new_state

    def init_state(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

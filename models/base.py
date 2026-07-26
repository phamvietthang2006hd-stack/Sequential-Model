from abc import ABC, abstractmethod
from typing import TypeAlias

import torch
from torch import Tensor, nn

RecurrentState: TypeAlias = Tensor | tuple[Tensor, Tensor]

class RecurrentCell(nn.Module, ABC):
    """Interface for a single-step recurrent cell."""

    input_dim: int
    hidden_dim: int

    @abstractmethod
    def forward(
        self, x_t: Tensor, state: RecurrentState
    ) -> tuple[Tensor, RecurrentState]:
        """
        Compute one recurrent transition

        Args:
            x_t: Input at timestep ``t``, shape ``(batch_size, input_dim)``.
            state: Previous state. RNN/GRU use ``h_{t-1}``; LSTM uses
                ``(h_{t-1}, c_{t-1})``.

        Returns:
            The current hidden state and the updated recurrent state.
        """

        raise NotImplementedError

    @abstractmethod
    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> RecurrentState:
        """Initialize a batch state on the requested device and dtype."""

        raise NotImplementedError

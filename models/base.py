# ./base.py

from abc import ABC, abstractmethod
import torch 
from torch import Tensor, nn

class RecurrentCell(nn.Module, ABC):

    @abstractmethod
    def forward(self, x_t: Tensor, hidden_state: Tensor | tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """
        Compute one recurrent transition

        @params:
            x_t: Input at timestep t, shape (batch_size, input_dim)
            hidden_state : Previous recurrent state
                - RNN/GRU use h_{t-1}
                - LSTM use (h_{t-1}, c_{t-1})
        return:
            h_t: next hidden_state
            new_state: Updated recurrent state:
                - timestep t + 1
                - RNN/GRUL h_t
                - LSTM: (h_t, c_t)
        """

        raise NotImplementedError

    @abstractmethod
    def init_state(self, batch_size: int, device: torch.device) -> Tensor | tuple[Tensor, Tensor]:
        """
        Initialize the recurrent state for a batch
        """

        raise NotImplementedError

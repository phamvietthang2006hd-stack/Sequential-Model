from torch import Tensor, nn

from .base import RecurrentCell

class SequenceModel(nn.Module):

    def __init__(self, cell: RecurrentCell, hidden_dim: int, output_dim: int):
        super().__init__()
        self.cell = cell
        if getattr(cell, "hidden_dim", hidden_dim) != hidden_dim:
            raise ValueError("hidden_dim must match cell.hidden_dim")
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Process a sequence and return final prediction
        @params:
            x: Input sequence, shape (batch_size, seq_len, input_dim)
        
        return:
            Prediction, shape (batchsize, output_dim)
        """

        if x.ndim != 3:
            raise ValueError("x must have shape (batch_size, seq_len, input_dim)")
        if x.size(1) == 0:
            raise ValueError("x must contain at least one timestep")

        batch_size = x.size(0)
        state = self.cell.init_state(batch_size=batch_size, device=x.device)
        for t in range(x.size(1)):
            h_t, state = self.cell(x[:,t,:], state)

        return self.output_layer(h_t)

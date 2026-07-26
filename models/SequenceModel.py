from torch import Tensor, nn

from .base import RecurrentCell


class SequenceModel(nn.Module):
    """Sequence-to-vector wrapper for a :class:`RecurrentCell`."""

    def __init__(self, cell: RecurrentCell, hidden_dim: int, output_dim: int):
        super().__init__()
        if not isinstance(cell, RecurrentCell):
            raise TypeError("cell must implement RecurrentCell")
        if hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("hidden_dim and output_dim must be positive")
        self.cell = cell
        if cell.hidden_dim != hidden_dim:
            raise ValueError("hidden_dim must match cell.hidden_dim")
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Process a sequence and return final prediction
        Args:
            x: Input sequence, shape ``(batch_size, seq_len, input_dim)``.

        Returns:
            Prediction for each sequence, shape ``(batch_size, output_dim)``.
        """

        if x.ndim != 3:
            raise ValueError("x must have shape (batch_size, seq_len, input_dim)")
        if x.size(1) == 0:
            raise ValueError("x must contain at least one timestep")
        if x.size(2) != self.cell.input_dim:
            raise ValueError("x.size(2) must match cell.input_dim")

        batch_size = x.size(0)
        state = self.cell.init_state(
            batch_size=batch_size, device=x.device, dtype=x.dtype
        )
        for t in range(x.size(1)):
            h_t, state = self.cell(x[:, t, :], state)

        return self.output_layer(h_t)

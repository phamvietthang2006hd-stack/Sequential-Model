import torch
from models.SequenceModel import SequenceModel
from models.GRU import GRUCell

def test_gru_cell_forward() -> None:
    """Test GRU Cell forward"""
    batch_size = 8
    input_dim = 10
    hidden_dim = 32

    cell = GRUCell(input_dim=input_dim, hidden_dim=hidden_dim)
    x = torch.randn(input_dim, hidden_dim)
    state = cell.init_state(batch_size=batch_size, device=x.device)

    h, new_state = cell.forward(x, state)

    assert new_state.shape == (batch_size, hidden_dim)

def test_gru_sequence_forward() -> None:
    """Test GRU Sequence forward"""
    batch_size = 4
    sequence_len = 20
    input_dim = 10
    hidden_dim = 64
    output_dim = 1

    cell = GRUCell(input_dim=input_dim, hidden_dim=hidden_dim)
    model = SequenceModel(cell, hidden_dim, output_dim) 
    x = torch.randn(batch_size, sequence_len, input_dim)

    pred = model.forward(x)

    assert pred.shape == (batch_size, output_dim)

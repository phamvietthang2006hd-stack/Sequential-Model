import torch 
from models.LSTM import LSTMCell
from models.SequenceModel import SequenceModel

def test_lstm_cell_forward() -> None:
    """Test LSTM cell forward"""
    batch_size = 8
    input_dim = 10
    hidden_dim = 32

    cell = LSTMCell(input_dim=input_dim, hidden_dim=hidden_dim)
    x = torch.randn(batch_size, hidden_dim)
    state = cell.init_state(batch_size=batch_size, device=x.device)

    h, new_state = cell.forward(x, state)

    h_state, c_state = new_state

    assert h.shape == (batch_size, hidden_dim)
    assert h_state.shape == (batch_size, hidden_dim)
    assert c_state.shape == (batch_size, hidden_dim)

def test_lstm_sequence_forward() -> None:
    """Test LSTM Sequence Forward"""
    batch_size = 4
    seq_length = 20
    input_dim = 10
    hidden_dim = 64
    output_dim = 1

    cell = LSTMCell(input_dim, hidden_dim)
    model = SequenceModel(cell, hidden_dim, output_dim)
    x = torch.randn(batch_size, seq_length, input_dim)

    pred = model.forward(x)

    assert pred.shape == (batch_size, output_dim)

import torch
from models.RNN import RNNCell
from models.SequenceModel import SequenceModel

def test_rnn_cell_forward() -> None:
    """Test RNN Cell forward"""
    INPUT_DIM = 10
    BATCH_SIZE = 8
    HIDDEN_DIM = 32

    cell = RNNCell(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
    x = torch.randn(BATCH_SIZE, INPUT_DIM)

    state = cell.init_state(batch_size=BATCH_SIZE, device=x.device)
    h, new_state = cell(x, state)

    assert h.shape == (BATCH_SIZE, HIDDEN_DIM)
    assert new_state.shape == (BATCH_SIZE, HIDDEN_DIM)

def test_rnn_sequence_forward() -> None:
    """Test RNN Sequence Model"""
    SEQUENCE_LEN = 20
    BATCH_SIZE = 4
    INPUT_DIM = 10
    HIDDEN_DIM = 64
    OUTPUT_DIM = 1

    cell = RNNCell(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
    model = SequenceModel(cell=cell, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
    x = torch.randn(BATCH_SIZE, SEQUENCE_LEN, INPUT_DIM)

    pred = model.forward(x)

    assert pred.shape == (BATCH_SIZE, OUTPUT_DIM)
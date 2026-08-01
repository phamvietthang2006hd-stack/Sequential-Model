import torch
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader

from data.dataset import SequenceDataset
from models.RNN import RNNCell
from models.SequenceModel import SequenceModel
from training.losses import MSELoss
from training.trainer import Trainer

def main() -> None:
    INPUT_DIM = 10
    HIDDEN_DIM = 64
    OUTPUT_DIM = 1

    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 50

    train_dataset = SequenceDataset(train=True)
    valid_dataset = SequenceDataset(train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    cell = RNNCell(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)

    criterion = MSELoss()

    model = SequenceModel(cell=cell, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainer = Trainer(
        model=model, 
        optimizer=optimizer, 
        criterion=criterion,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    trainer.fit(train_loader=train_loader, validation_loader=valid_loader, epochs=EPOCHS)

if __name__ == '__main__':
    main()
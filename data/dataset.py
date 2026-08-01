from torch import nn, Tensor
from torch.utils.data import Dataset
import torch
import numpy as np

class SequenceDataset(Dataset):
    """Synthetic sequence dataset"""
    def __init__(self, seq_length: int = 20, train: bool = True, dataset_size: int = 1000) -> None:
        self.seq_length = seq_length

        np.random(42)
        t = np.linspace(0, 100, dataset_size + seq_length)
        signal = np.sin(t)

        inputs = []
        targets = []

        for i in range(dataset_size):
            x = signal[i : i + seq_length]
            y = signal[i + seq_length]

            inputs.append(x)
            targets.append(y)

        inputs = np.array(inputs)
        targets = np.array(targets)

        split = int(0.8 * dataset_size)

        if train:
            self.x = inputs[:split]
            self.y = inputs[:split]
        else:
            self.x = inputs[split:]
            self.y = inputs[split:]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        x = torch.tensor(self.x[index], dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(self.y[index], dtype=torch.float32).unsqueeze(0)

        return x, y

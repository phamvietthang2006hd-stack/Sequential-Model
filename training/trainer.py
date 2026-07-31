import torch
from torch import Tensor,nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

class Trainer:
    """Generic trainer"""
    def __init__(self, model: nn.Module, optimizer: Optimizer, criterion: nn.Module, device: str = 'cpu') -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_step(self, x: Tensor, y: Tensor) -> float:
        self.model.train()

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        self.optimizer.zero_grad()

        pred = self.model(x)

        loss = self.criterion(pred, y)

        loss.backward()

        self.optimizer.step()

        return loss.item()         

    @torch.no_grad()
    def validate_step(self, x: Tensor, y: Tensor) -> float:
        self.model.eval()

        x = x.to(self.device)
        y = y.to(self.device)

        prediction = self.model(x)

        loss = self.criterion(prediction, y)
        return loss.item()

    def train_epoch(self, dataloader: DataLoader) -> float:
        total_loss = 0.0

        for x,y in dataloader:
            total_loss += self.train_step(x, y)

        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader) -> float:
        total_loss = 0.0

        for x, y in dataloader:
            total_loss += self.validate_step(x,y)

        return total_loss / len(dataloader)

    def fit(self, train_loader: DataLoader, validation_loader: DataLoader, epochs: int) -> None:
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            valid_loss = self.validate_epoch(validation_loader)
            print(
                f"Epoch {epoch}, Train: {train_loss:.6f}, Valid: {valid_loss:.6f}"
            )

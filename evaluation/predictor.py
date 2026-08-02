import torch
from torch import Tensor
from torch.utils.data import DataLoader
from models.SequenceModel import SequenceModel

class Predictor:
    def __init__(self, model: SequenceModel, device: str = 'cpu') -> None:
        self.model = model 
        self.device = device

    @torch.no_grad
    def predict(self, dataloader: DataLoader) -> tuple[Tensor, Tensor]:
        self.model.eval()
        predictor = []
        targets = []

        for x, y in dataloader:
            x = x.to(self.device) 
            y = y.to(self.device)

            pred = self.model(x)

            predictor.append(pred.cpu())
            targets.append(y.cpu())

        prediction = torch.cat(predictor,dim=0)
        target = torch.cat(targets, dim=0)

        return prediction, target

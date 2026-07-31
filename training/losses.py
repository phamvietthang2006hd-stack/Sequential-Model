from torch import Tensor, nn

class MSELoss(nn.Module):
    """Mean Square Error loss - Regression"""
    def __init__(self) -> None:
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return nn.loss(prediction, target)

class CrossEntropyLoss(nn.Module):
    """Cross Entropy Loss - Classification"""
    def __init__(self) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, prediction: None, target: None) -> Tensor:
        return self.loss(prediction, target)
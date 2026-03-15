import torchvision
from torch import nn, Tensor


class Falkenstein(nn.Module):
    """A DenseNet121-based classification model implemented to work on the CUB-200-dataset.
    The classification-layer is replaced by a custom layer, containing a Linear layer (1024,512) and ReLU,
    and a Linear output layer with (512,200).
    Weights of the last 2 DenseBlocks are unfreezed, while others are freezed."""

    def __init__(self):
        super().__init__()
        self.denseNet121 = torchvision.models.densenet121(
            weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
        )

        for name, param in self.denseNet121.named_parameters():
            if any(
                x in name
                for x in ["denseblock3", "transition3", "denseblock4", "norm5"]
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.denseNet121.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 200),
        )

    def forward(self, X: Tensor) -> Tensor:
        """A forward pass of the network.

        Args:
            X (Tensor): The data used in the forward pass

        Returns:
            Tensor: The result of the forward pass
        """

        return self.denseNet121(X)

import torchvision
from torch import nn, Tensor


class Falkenstein(nn.Module):
    """A DenseNet121-based classification model implemented to work on the CUB-200-dataset. The classification-layer is replaced by a custom layer.

    """    

    def __init__(self):        
        super().__init__()
        self.denseNet121 = torchvision.models.densenet121(pretrained=torchvision.models.DenseNet121_Weights.DEFAULT)
        for param in self.denseNet121.parameters():
            param.requires_grad = False
        self.denseNet121.classifier = nn.Linear(in_features=1024, out_features=200)


    def forward(self, X: Tensor):
        """A forward pass of the network.

        Args:
            X (Tensor): The data used in the forward pass
        """        
        
        return self.denseNet121(X)
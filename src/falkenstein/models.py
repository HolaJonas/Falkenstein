import torchvision
from torch import nn, Tensor


class Falkenstein(nn.Module):
    """A DenseNet121-based classification model implemented to work on the CUB-200-dataset.
    The classification-layer is replaced by either a mlp or a linear layer, based on parameters.

    Args:
        output_layers_type (str): The type of classifier layers. 'linear' or 'mlp'. Defaults to 'mlp'.
        dropout (float): The dropout rate of each layer. Defaults to 0.
        classifier_hidden_dim (int): The number of hidden neurons in the mlp. Only applies when 'output_layers_type' is 'mlp'. Defaults to 512.

    """

    def __init__(
        self,
        output_layers_type: str = "mlp",
        dropout: float = 0,
        classifier_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.denseNet121 = torchvision.models.densenet121(
            weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
        )

        for name, param in self.denseNet121.named_parameters():
            if any(
                x in name
                for x in [
                    "denseblock3",
                    "transition3",
                    "denseblock4",
                    "norm5",
                    "transition2",
                    "denseblock2",
                    "transition1",
                    "denseblock1",
                ]
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False

        if output_layers_type == "linear":
            self.denseNet121.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(1024, 200),
            )
        elif output_layers_type == "mlp":
            self.denseNet121.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(1024, classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, 200),
            )
        else:
            raise ValueError("Unsupported output_layer_type. 'linear' or 'mlp'.")

    def forward(self, X: Tensor) -> Tensor:
        """A forward pass of the network.

        Args:
            X (Tensor): The data used in the forward pass

        Returns:
            Tensor: The result of the forward pass
        """

        return self.denseNet121(X)

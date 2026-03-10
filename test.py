import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy


def test(model: nn.Module, dataloader: DataLoader, device: torch.device) -> None:
    """Evalutes performance of a pytorch model on a provided dataset.

    Args:
        model (nn.Module): The model to be evaluated
        dataloader (DataLoader): The data being evaluated on
        device (torch.device): The device to run evaluation on
    """

    test_labels_gt = []
    test_labels_pred = []
    model.eval()
    with torch.no_grad():
        for input, label in dataloader:
            input = input.to(device)
            label = label.to(device)
            y_hat_probs = model(input)
            test_labels_pred.append(torch.argmax(y_hat_probs, dim=1))
            test_labels_gt.append(label)

    preds = torch.cat(test_labels_pred)
    targets = torch.cat(test_labels_gt)
    accuracy = Accuracy(task="multiclass", num_classes=200).to(device)
    print(accuracy(preds, targets))

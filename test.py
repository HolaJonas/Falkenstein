import torch
import numpy as np
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from sklearn.metrics import top_k_accuracy_score


def test(model: nn.Module, dataloader: DataLoader, device: torch.device) -> None:
    """Evalutes performance of a pytorch model on a provided dataset.

    Args:
        model (nn.Module): The model to be evaluated
        dataloader (DataLoader): The data being evaluated on
        device (torch.device): The device to run evaluation on
    """

    test_labels_gt = []
    test_labels_pred = []
    y_score = []
    model.eval()
    with torch.no_grad():
        for input, label in dataloader:
            input = input.to(device)
            label = label.to(device)
            y_hat_probs = model(input)
            y_score.append(Tensor.cpu(y_hat_probs).detach().numpy())
            test_labels_pred.append(torch.argmax(y_hat_probs, dim=1))
            test_labels_gt.append(label)

    preds = torch.cat(test_labels_pred)
    targets = torch.cat(test_labels_gt)
    accuracy = Accuracy(task="multiclass", num_classes=200).to(device)
    print(accuracy(preds, targets))
    y_score_all = np.concatenate(y_score, axis=0)
    targets_np = Tensor.cpu(targets).detach().numpy()
    print(top_k_accuracy_score(targets_np, y_score_all, k=5, labels=list(range(200))))

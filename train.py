from torch import nn, optim
from torch.utils.data import DataLoader
import tqdm
import torch


def train(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
) -> None:
    """Trains a provided pytorch model on provided data.

    Args:
        model (nn.Module): The model to be trained
        dataloader (DataLoader): The training data
        criterion (nn.Module): The loss function to be trained with
        optimizer (optim.Optimizer): The optimizer used to train
        num_epochs (int): The number of iterations on the whole dataset
    """

    for epoch in tqdm.tqdm(range(num_epochs)):
        model.train()
        for input, label in dataloader:
            y_hat = model(input.to(device))
            loss = criterion(y_hat, label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} of {num_epochs + 1}")

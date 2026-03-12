from torch import nn, optim
from torch.utils.data import DataLoader
import tqdm
import torch
import copy
from math import inf
from src.falkenstein.utils.plot import plot_loss
from torch.optim import lr_scheduler


def train(
    model: nn.Module,
    train_data: DataLoader,
    validation_data: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    patience: int = 5,
    scheduler: lr_scheduler.LRScheduler | None = None,
) -> None:
    """Trains a provided pytorch model on provided data with early stopping.

    Args:
        model (nn.Module): The model to be trained
        train_data (DataLoader): The training data
        validation_data (DataLoader): The validation data
        criterion (nn.Module): The loss function to be trained with
        optimizer (optim.Optimizer): The optimizer used to train
        num_epochs (int): Maximum number of training epochs
        device (torch.device): Device to train on
        patience (int): Number of epochs without improvement before stopping. Defaults to 5.
    """

    best_validation_loss = inf
    validation_loss_mem = []
    training_loss_mem = []
    epochs_no_improvement = 0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in tqdm.tqdm(range(num_epochs)):
        training_loss = 0.0
        model.train()
        for input, label in train_data:
            y_hat = model(input.to(device))
            loss = criterion(y_hat, label.to(device))
            training_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        training_loss /= len(train_data)
        training_loss_mem.append(training_loss)

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for input, label in validation_data:
                y_hat = model(input.to(device))
                validation_loss += criterion(y_hat, label.to(device)).item()
        validation_loss /= len(validation_data)
        validation_loss_mem.append(validation_loss)

        print(
            f"Epoch {epoch + 1} of {num_epochs}, training loss: {training_loss}, validation loss: {validation_loss}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_no_improvement = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= patience:
                print(f"Early stopping, epoch {epoch + 1}.")
                break

    model.load_state_dict(best_weights)
    plot_loss(validation_loss_mem, "validation loss")
    plot_loss(training_loss_mem, "training loss")
    print(f"Best validation loss: {best_validation_loss}")

from torch import nn
from torch.optim import AdamW, lr_scheduler
from src.falkenstein.models import Falkenstein
from src.falkenstein.data import generate_dataset, create_dataloaders
from train import train
from test import test
import yaml
from torch.cuda.memory import set_per_process_memory_fraction
import torch
import random
import numpy as np
from datetime import datetime


def seed() -> None:
    """Seeds all used imports to make results reproducable."""

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


def init_cuda(fraction: float = 0.33):
    """Initializes all necessary cuda settings to limit capacity.

    Returns:
        Device: _description_
    """    
    set_per_process_memory_fraction(fraction=fraction)
    torch.set_num_threads(3)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    seed()
    device = init_cuda(0.33)

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    dataset = generate_dataset("data/CUB_200_2011/images")
    train_data, validate_data, test_data = create_dataloaders(
        dataset=dataset,
        train_split=config["train_split"],
        validation_split=config["validation_split"],
        test_split=config["test_split"],
        batch_size=config["batch_size"],
    )

    model = Falkenstein().to(device=device)
    optimizer = AdamW(params=model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    train(
        model=model,
        dataloader=train_data,
        validation_data=validate_data,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config["num_epochs"],
        device=device,
        patience=config["patience"],
        scheduler=scheduler
    )
    torch.save(model.state_dict(), f"weights/{datetime.now()}")
    test(model=model, dataloader=train_data, device=device)
    test(model=model, dataloader=test_data, device=device)

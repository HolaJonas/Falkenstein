from torch import nn
from torch.optim import Adam
from src.falkenstein.models import Falkenstein
from src.falkenstein.data import generate_dataset, create_dataloaders
from train import train
from test import test
import yaml
from torch.cuda.memory import set_per_process_memory_fraction
import torch
import random
import numpy as np


def seed() -> None:
    """Seeds all used imports to make results reproducable."""

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


if __name__ == "__main__":
    seed()
    set_per_process_memory_fraction(fraction=0.33)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    optimizer = Adam(params=model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    train(
        model=model,
        dataloader=train_data,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config["num_epochs"],
        device=device
    )
    test(model=model, dataloader=test_data, device=device)

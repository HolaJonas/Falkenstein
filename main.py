from torch import nn
from torch.optim import AdamW, lr_scheduler
from src.falkenstein.models import Falkenstein
from src.falkenstein.data import generate_dataset, create_dataloaders
from src.falkenstein.train import train
from src.falkenstein.test import test
import yaml
from torch.cuda.memory import set_per_process_memory_fraction
import torch
import random
import numpy as np
from datetime import datetime
import os
import sys
from src.falkenstein.utils.logger import Tee


def __seed() -> None:
    """Seeds all used imports to make results reproducable."""

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


def __init_cuda(fraction: float = 0.33) -> torch.device:
    """Initializes all necessary cuda settings to limit capacity.

    Returns:
        torch.device: The device used for training
    """

    set_per_process_memory_fraction(fraction=fraction)
    torch.set_num_threads(3)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model(config: str, cub_200_2001_path: str) -> None:
    """Trains, validates and tests a DenseNet121-based classification model on CUB-200-2011.


    Args:
        config (str): The path of a yaml config containing:
            'train_split',
            'validation_split',
            'test_split',
            'learning_rate',
            'label_smoothing',
            'patience',
            'num_epochs',
            'batch_size',
            'weight_decay'
            'classifier_hidden_dim'
            'output_layers_type'
            'dropout'

        cub_200_2001_path (str): The path to the dataset
    """

    __seed()
    device = __init_cuda(0.33)

    with open(config) as f:
        config = yaml.safe_load(f)

    dataset = generate_dataset(cub_200_2001_path)
    train_data, validate_data, _ = create_dataloaders(
        dataset=dataset,
        train_split=config["train_split"],
        validation_split=config["validation_split"],
        test_split=config["test_split"],
        batch_size=config["batch_size"],
        device=device,
    )

    model = Falkenstein(
        dropout=config["dropout"],
        head_type=config["output_layers_type"],
        classifier_hidden_dim=config["classifier_hidden_dim"],
    ).to(device=device)
    optimizer = AdamW(
        params=model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    train(
        model=model,
        train_data=train_data,
        validation_data=validate_data,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config["num_epochs"],
        device=device,
        patience=config["patience"],
        scheduler=scheduler,
    )
    torch.save(model.state_dict(), f"weights/{datetime.now()}")
    test(model=model, dataloader=train_data, device=device)
    test(model=model, dataloader=validate_data, device=device)


if __name__ == "__main__":

    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/{datetime.now()}.log"
    log_file = open(log_path, "w")

    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    model("configs/config.yaml", "data/CUB_200_2011/images")

from matplotlib import pyplot as plt
from numpy import argmin
from datetime import datetime


def plot_loss(train_loss_data: list, validation_loss_data: list) -> None:
    """Plots the loss against the epochs.

    Args:
        train_loss_data (list): The training losses in chronological order
        validation_loss_data (list): The validation losses in chronological order
    """    
    
    fig, ax = plt.subplots()
    ax.plot(range(0, len(train_loss_data)), train_loss_data, c="red", label="train")
    ax.plot(range(0, len(validation_loss_data)), validation_loss_data, c="blue", label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    min_idx = int(argmin(validation_loss_data))
    min_val = validation_loss_data[min_idx]
    ax.annotate(f"{min_val:.4f}", (min_idx, min_val))
    ax.legend()
    fig.savefig(f"results/loss/losses_{datetime.now()}.png")
    plt.close(fig)
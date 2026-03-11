from matplotlib import pyplot as plt
from numpy import argmax
from datetime import datetime


def plot_loss(loss_data: list, loss_type: str) -> None:
    """Plots the loss against the epochs.

    Args:
        loss_data (list): An array of all stored losses in chronological order
        loss_type (str): The loss type, used as label
    """    
    
    fig, ax = plt.subplots()
    ax.plot(range(0, len(loss_data)), loss_data)
    ax.set_xlabel("epoch")
    ax.set_ylabel(loss_type)
    ax.annotate(f"{max(loss_data):.4f}", (argmax(loss_data), max(loss_data)))
    fig.savefig(f"results/{loss_type.replace(' ', '_')}_{datetime.now()}.png")
    plt.close(fig)
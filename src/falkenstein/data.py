from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    Compose,
    CenterCrop,
    ToTensor,
    Normalize,
    Resize,
    RandomRotation,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ColorJitter,
)
from torch import Tensor, nn
from torch.utils.data import random_split, DataLoader, Dataset
from PIL import Image
import tarfile
import os
import torch


def prepare_dataset(file_path: str, extract_path: str) -> None:
    """Unpacks the tgz file of the dataset.

    Args:
        file_path (str): The path of the tgz file
        extract_path (str): The path to store the unpacked tgz
    """
        
    os.makedirs("data", exist_ok=True)
    file = tarfile.open(file_path, "r")
    for item in file:
        file.extract(item, extract_path)


class AugmentedData(Dataset):
    """Implements a custom dataset with optinal transformations. Applies default transformations:
        Resize(224), CenterCrop(224), ToTenser(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    Args:
        dataset (Dataset): The dataset
        transforms (nn.Module): The transforms applied to the dataset
    """

    transforms = Compose(
        [
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(self, dataset: Dataset, transforms: nn.Module | None = None):
        self.data = dataset
        self.transforms = transforms

    def __len__(self) -> int:
        """Number of elements in the dataset.

        Returns:
            int: The number of elements in the dataset
        """

        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Returns the imagedata at given index.

        Args:
            index (int): The index of the imagedata

        Returns:
            tuple[Tensor, int]: The image as a Tensor and its according label
        """

        img, label = self.data.dataset.imgs[self.data.indices[index]]
        img = Image.open(img).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = AugmentedData.transforms(img)
        return img, label


def generate_dataset(path: str) -> ImageFolder:
    """Initializes the dataset from path.

    Args:
        path (str): The path to the dataset

    Returns:
        ImageFolder: The prepared dataset.
    """

    dataset = ImageFolder(root=path, transform=None)
    return dataset


def create_dataloaders(
    dataset: ImageFolder,
    device: torch.device,
    train_split: float = 0.5,
    test_split: float = 0.25,
    validation_split: float = 0.25,
    batch_size: int = 256,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Splits the dataset into train-, validation- and test-data. Establishes batches.

    Args:
        dataset (ImageFolder): The dataset
        train_split (float, optional): The train split in percent. Defaults to 0.5.
        test_split (float, optional): The test split in percent. Defaults to 0.25.
        validation_split (float, optional): The validatiom split in percent. Defaults to 0.25.
        batch_size (int, optional): The batch size. Defaults to 64.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: A triple containing train-, validation- and testdata
    """

    train, validate, test = random_split(
        dataset=dataset,
        lengths=[train_split, validation_split, test_split],
        generator=torch.Generator(device=device).manual_seed(0),
    )

    train = AugmentedData(
        train,
        transforms=Compose(
            [
                RandomResizedCrop(224, scale=(0.6, 1.0)),
                RandomHorizontalFlip(),
                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                RandomRotation(20),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )
    test = AugmentedData(test)
    validate = AugmentedData(validate)

    train_dataloader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_dataloader = DataLoader(
        test, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        validate, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    return train_dataloader, val_dataloader, test_dataloader

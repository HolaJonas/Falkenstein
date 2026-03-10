from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize
from torch.utils.data import random_split, DataLoader


def generate_dataset(path: str) -> ImageFolder:
    """Initializes the dataset from path.

    Args:
        path (str): The path to the dataset

    Returns:
        ImageFolder: The prepared dataset.
    """

    data_transforms = Compose(
        [
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ImageFolder(root=path, transform=data_transforms)
    return dataset


def create_dataloaders(
    dataset: ImageFolder,
    train_split: float = 0.6,
    test_split: float = 0.2,
    validation_split: float = 0.2,
    batch_size: int = 10,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Splits the dataset into train-, validation- and test-data. Establishes batches.

    Args:
        dataset (ImageFolder): The dataset
        train_split (float, optional): The train split in percent. Defaults to 0.6.
        test_split (float, optional): The test split in percent. Defaults to 0.2.
        validation_split (float, optional): The validatiom split in percent. Defaults to 0.2.
        batch_size (int, optional): The batch size. Defaults to 10.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: A triple containing train-, validation- and testdata
    """

    train, validate, test = random_split(
        dataset=dataset, lengths=[train_split, validation_split, test_split]
    )
    train_dataloader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_dataloader = DataLoader(
        validate, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        test, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    return train_dataloader, val_dataloader, test_dataloader

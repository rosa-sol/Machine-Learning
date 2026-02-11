import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_pcam_loaders(config):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder("pcam/train", transform=transform)
    val_dataset = datasets.ImageFolder("pcam/val", transform=transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=config["batch_size"],
                              shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=config["batch_size"],
                            shuffle=False)

    return train_loader, val_loader

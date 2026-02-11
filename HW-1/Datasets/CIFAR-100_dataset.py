import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def get_cifar100_dataloaders(batch_size):
    # Transforms (reused from Oh5m1BhiftJi)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-100 training data
    full_trainset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Load test data
    testset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Split training into train + validation (80/20 split)
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size

    train_dataset, val_dataset = random_split(
        full_trainset,
        [train_size, val_size]
    )

    # Create loaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader, testloader

print("Generated content for dataset.py.")

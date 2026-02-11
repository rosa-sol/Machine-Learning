import os
import gzip
import shutil
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# --------------------------------------------------
# Decompress .gz files if needed
# --------------------------------------------------
def decompress_gz_file(compressed_path, decompressed_path):
    if os.path.exists(decompressed_path):
        return decompressed_path

    print(f"Decompressing {compressed_path} ...")
    with gzip.open(compressed_path, 'rb') as f_in:
        with open(decompressed_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print("Decompression complete.")
    return decompressed_path


# --------------------------------------------------
# PCAM Dataset Class
# --------------------------------------------------
class PCamDataset(Dataset):
    """
    Lazy-loading HDF5 dataset for PCAM.
    Compatible with CNN and MLP models.
    Uses CrossEntropyLoss label format (long, scalar).
    """

    def __init__(self, x_path, y_path):
        self.x_path = x_path
        self.y_path = y_path

        # open files once
        self.x_file = h5py.File(self.x_path, 'r')
        self.y_file = h5py.File(self.y_path, 'r')

        self.images = self.x_file['x']
        self.labels = self.y_file['y']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]        # H, W, C
        label = self.labels[idx]

        # Convert image → tensor, normalize, CHW
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Convert label for CrossEntropyLoss
        label = torch.tensor(label, dtype=torch.long).squeeze()

        return img, label


# --------------------------------------------------
# DataLoader creator
# --------------------------------------------------
def get_pcam_loaders(
        x_path,
        y_path,
        batch_size=64,
        train_frac=0.7,
        val_frac=0.15,
        seed=42,
        num_workers=0,
        decompress=True
):
    """
    Returns:
        trainloader, valloader, testloader
    """

    # decompress if needed
    if decompress and x_path.endswith(".gz"):
        x_path = decompress_gz_file(x_path, x_path[:-3])
    if decompress and y_path.endswith(".gz"):
        y_path = decompress_gz_file(y_path, y_path[:-3])

    dataset = PCamDataset(x_path, y_path)

    total_size = len(dataset)
    train_size = int(train_frac * total_size)
    val_size = int(val_frac * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    trainloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    valloader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    testloader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return trainloader, valloader, testloader

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class AdultDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_adult_loaders(config):

    df = pd.read_csv("adult.csv")  # adjust path

    X = df.drop("income", axis=1).values
    y = df["income"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_dataset = AdultDataset(X_train, y_train)
    val_dataset = AdultDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset,
                              batch_size=config["batch_size"],
                              shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=config["batch_size"],
                            shuffle=False)

    return train_loader, val_loader

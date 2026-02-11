import torch
import torch.nn as nn
import torch.optim as optim

from configs.adult_config import CONFIG
from datasets.adult_dataset import get_adult_loaders
from models.adult_model import AdultMLP
from train.trainer import train_model
from utils.plot_curves import plot_curves


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader = get_adult_loaders(CONFIG)

model = AdultMLP(CONFIG).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=CONFIG["learning_rate"])

history = train_model(model,
                      train_loader,
                      val_loader,
                      optimizer,
                      criterion,
                      device,
                      CONFIG["epochs"])

plot_curves(history)

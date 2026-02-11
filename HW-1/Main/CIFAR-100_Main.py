import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import config
from dataset import get_cifar100_dataloaders
from model import CIFAR100Net, MLPNet
from training import train_model
from evaluation import evaluate_model

# 1. Initialize history dictionary
history = {
    "cnn_train_loss": [],
    "cnn_train_acc": [],
    "cnn_train_f1": [],
    "cnn_val_loss": [],
    "cnn_val_acc": [],
    "cnn_val_f1": [],

    "mlp_train_loss": [],
    "mlp_train_acc": [],
    "mlp_train_f1": [],
    "mlp_val_loss": [],
    "mlp_val_acc": [],
    "mlp_val_f1": []
}

# 2. Get data loaders
trainloader, valloader, testloader = get_cifar100_dataloaders(config.batch_size)

# 3. Instantiate models and move to device
cnn = CIFAR100Net().to(config.device)
mlp = MLPNet().to(config.device)

# 4. Initialize criterion
criterion = nn.CrossEntropyLoss()

# 5. Create optimizers
cnn_optimizer = optim.Adam(cnn.parameters(), lr=config.learning_rate)
mlp_optimizer = optim.Adam(mlp.parameters(), lr=config.learning_rate)

# 6. Train models
print("Starting training...")
train_model(
    cnn_model=cnn,
    mlp_model=mlp,
    trainloader=trainloader,
    valloader=valloader,
    cnn_optimizer=cnn_optimizer,
    mlp_optimizer=mlp_optimizer,
    criterion=criterion,
    num_epochs=config.num_epochs,
    device=config.device,
    history=history
)
print("Training complete.")

# 7. Evaluate models
print("Starting evaluation...")
cnn_test_loss, cnn_test_acc, cnn_test_f1 = evaluate_model(cnn, testloader, criterion, config.device, is_mlp=False)
mlp_test_loss, mlp_test_acc, mlp_test_f1 = evaluate_model(mlp, testloader, criterion, config.device, is_mlp=True)
print("Evaluation complete.")

# 8. Print final test results
print("\nFinal Test Results")
print(f"CNN  | Acc: {cnn_test_acc:.4f} | F1: {cnn_test_f1:.4f}")
print(f"MLP  | Acc: {mlp_test_acc:.4f} | F1: {mlp_test_f1:.4f}")

# 9. Plotting results
plt.figure(figsize=(10, 5))
plt.plot(history["cnn_train_loss"], label="CNN Train Loss")
plt.plot(history["cnn_val_loss"], label="CNN Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("CNN Loss over Epochs")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history["mlp_train_loss"], label="MLP Train Loss")
plt.plot(history["mlp_val_loss"], label="MLP Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("MLP Loss over Epochs")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history["cnn_train_acc"], label="CNN Train Accuracy")
plt.plot(history["cnn_val_acc"], label="CNN Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("CNN Accuracy over Epochs")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history["mlp_train_acc"], label="MLP Train Accuracy")
plt.plot(history["mlp_val_acc"], label="MLP Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("MLP Accuracy over Epochs")
plt.show()


print("Generated content for main.py. The plots will be displayed when main.py is executed.")

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# Assuming model.py is in the same directory
from model import CIFAR100Net, MLPNet

def evaluate_model(model, testloader, criterion, device, is_mlp=False):
    model.eval()
    correct = 0
    total = 0
    preds, labels_list = [], []
    loss_total = 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            if is_mlp:
                images = images.view(images.size(0), -1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_total += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            preds.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(labels_list, preds, average="macro")

    return loss_total / len(testloader), accuracy, f1

print("Generated content for evaluation.py.")

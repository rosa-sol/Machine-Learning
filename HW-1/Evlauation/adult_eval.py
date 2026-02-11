import torch
from sklearn.metrics import f1_score

def evaluate_adult(model, dataloader, criterion, device):
    model.eval()
    correct, total = 0, 0
    preds, labels_list = [], []
    loss_total = 0.0

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss_total += criterion(outputs, labels).item()

            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            preds.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(labels_list, preds, average="macro")
    avg_loss = loss_total / len(dataloader)
    return avg_loss, accuracy, f1

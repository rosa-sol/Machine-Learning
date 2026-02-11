import torch
from sklearn.metrics import f1_score

def evaluate_pcam(model, dataloader, criterion, device, is_mlp=False):
    model.eval()
    correct, total = 0, 0
    preds, labels_list = [], []
    loss_total = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            if is_mlp:
                images = images.view(images.size(0), -1)

            outputs = model(images)
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
"""
from eval.cifar_eval import evaluate_cifar

loss, acc, f1 = evaluate_cifar(cnn, testloader, criterion, device, is_mlp=False)
loss, acc, f1 = evaluate_cifar(mlp, testloader, criterion, device, is_mlp=True)
"""

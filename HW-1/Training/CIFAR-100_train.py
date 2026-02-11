import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import f1_score

# Assuming model.py and config.py are in the same directory
from model import CIFAR100Net, MLPNet
import config

def train_model(cnn_model, mlp_model, trainloader, valloader, cnn_optimizer, mlp_optimizer, criterion, num_epochs, device, history):
    for epoch in range(num_epochs):
        start_time = time.time()

        # =========================
        # ======== TRAIN ==========
        # =========================

        cnn_model.train()
        mlp_model.train()

        cnn_train_loss = 0.0
        cnn_correct = 0
        cnn_total = 0
        cnn_preds, cnn_labels = [], []

        mlp_train_loss = 0.0
        mlp_correct = 0
        mlp_total = 0
        mlp_preds, mlp_labels = [], []

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            # ----- CNN -----
            cnn_optimizer.zero_grad()
            outputs = cnn_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            cnn_optimizer.step()

            cnn_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            cnn_total += labels.size(0)
            cnn_correct += (predicted == labels).sum().item()
            cnn_preds.extend(predicted.cpu().numpy())
            cnn_labels.extend(labels.cpu().numpy())

            # ----- MLP -----
            mlp_optimizer.zero_grad()
            flat_images = images.view(images.size(0), -1)
            outputs = mlp_model(flat_images)
            loss = criterion(outputs, labels)
            loss.backward()
            mlp_optimizer.step()

            mlp_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            mlp_total += labels.size(0)
            mlp_correct += (predicted == labels).sum().item()
            mlp_preds.extend(predicted.cpu().numpy())
            mlp_labels.extend(labels.cpu().numpy())

        cnn_train_acc = cnn_correct / cnn_total
        cnn_train_f1 = f1_score(cnn_labels, cnn_preds, average="macro")

        mlp_train_acc = mlp_correct / mlp_total
        mlp_train_f1 = f1_score(mlp_labels, mlp_preds, average="macro")

        # =========================
        # ===== VALIDATION ========
        # =========================

        cnn_model.eval()
        mlp_model.eval()

        cnn_val_loss = 0.0
        cnn_correct = 0
        cnn_total = 0
        cnn_preds, cnn_labels = [], []

        mlp_val_loss = 0.0
        mlp_correct = 0
        mlp_total = 0
        mlp_preds, mlp_labels = [], []

        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)

                # ----- CNN -----
                outputs = cnn_model(images)
                loss = criterion(outputs, labels)
                cnn_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                cnn_total += labels.size(0)
                cnn_correct += (predicted == labels).sum().item()
                cnn_preds.extend(predicted.cpu().numpy())
                cnn_labels.extend(labels.cpu().numpy())

                # ----- MLP -----
                flat_images = images.view(images.size(0), -1)
                outputs = mlp_model(flat_images)
                loss = criterion(outputs, labels)
                mlp_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                mlp_total += labels.size(0)
                mlp_correct += (predicted == labels).sum().item()
                mlp_preds.extend(predicted.cpu().numpy())
                mlp_labels.extend(labels.cpu().numpy())

        cnn_val_acc = cnn_correct / cnn_total
        cnn_val_f1 = f1_score(cnn_labels, cnn_preds, average="macro")

        mlp_val_acc = mlp_correct / mlp_total
        mlp_val_f1 = f1_score(mlp_labels, mlp_preds, average="macro")

        # =========================
        # ===== STORE METRICS =====
        # =========================

        history["cnn_train_loss"].append(cnn_train_loss / len(trainloader))
        history["cnn_train_acc"].append(cnn_train_acc)
        history["cnn_train_f1"].append(cnn_train_f1)
        history["cnn_val_loss"].append(cnn_val_loss / len(valloader))
        history["cnn_val_acc"].append(cnn_val_acc)
        history["cnn_val_f1"].append(cnn_val_f1)

        history["mlp_train_loss"].append(mlp_train_loss / len(trainloader))
        history["mlp_train_acc"].append(mlp_train_acc)
        history["mlp_train_f1"].append(mlp_train_f1)
        history["mlp_val_loss"].append(mlp_val_loss / len(valloader))
        history["mlp_val_acc"].append(mlp_val_acc)
        history["mlp_val_f1"].append(mlp_val_f1)

        epoch_time = time.time() - start_time

        print(f"""
Epoch [{epoch+1}/{num_epochs}]
CNN  | Train Acc: {cnn_train_acc:.4f} | Val Acc: {cnn_val_acc:.4f}
MLP  | Train Acc: {mlp_train_acc:.4f} | Val Acc: {mlp_val_acc:.4f}
Time: {epoch_time:.2f}s
""")

print("Generated content for training.py.")

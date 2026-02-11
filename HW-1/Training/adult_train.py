import torch
import time
from sklearn.metrics import f1_score


def train_model(model, train_loader, val_loader,
                optimizer, criterion, device, epochs):

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": []
    }

    total_start = time.time()

    for epoch in range(epochs):
        start = time.time()

        # TRAIN
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(y.cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = sum(
            [p == t for p, t in zip(train_preds, train_labels)]
        ) / len(train_labels)

        train_f1 = f1_score(train_labels, train_preds, average="weighted")

        # VALIDATION
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)

                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = sum(
            [p == t for p, t in zip(val_preds, val_labels)]
        ) / len(val_labels)

        val_f1 = f1_score(val_labels, val_preds, average="weighted")

        epoch_time = time.time() - start

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss {train_loss:.4f} | "
              f"Val Loss {val_loss:.4f} | "
              f"Train Acc {train_acc:.4f} | "
              f"Val Acc {val_acc:.4f} | "
              f"Time {epoch_time:.2f}s")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

    total_time = time.time() - total_start
    print(f"\nTotal Training Time: {total_time:.2f} seconds")

    return history

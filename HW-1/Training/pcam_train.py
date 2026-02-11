import torch
import time
from sklearn.metrics import f1_score


def train_pcam(model,
               trainloader,
               valloader,
               optimizer,
               criterion,
               device,
               epochs=12,
               is_mlp=False):
    """
    Training loop specifically for PCAM dataset.

    Works for:
    - CNN (is_mlp=False)
    - MLP (is_mlp=True → flattens images)

    Tracks per epoch:
    - loss
    - accuracy
    - F1
    - execution time

    Returns:
        history dictionary for plotting
    """

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
        "epoch_time": []
    }

    total_start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # ===== TRAINING =====
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        preds, labels_list = [], []

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            if is_mlp:
                images = images.view(images.size(0), -1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            preds.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

        train_loss = running_loss / len(trainloader)
        train_acc = correct / total
        train_f1 = f1_score(labels_list, preds, average="macro")

        # ===== VALIDATION =====
        model.eval()
        val_loss_total = 0.0
        correct = 0
        total = 0
        preds, labels_list = [], []

        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)

                if is_mlp:
                    images = images.view(images.size(0), -1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item()

                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                preds.extend(predicted.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        val_loss = val_loss_total / len(valloader)
        val_acc = correct / total
        val_f1 = f1_score(labels_list, preds, average="macro")

        epoch_time = time.time() - epoch_start

        # Save metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["epoch_time"].append(epoch_time)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

    total_training_time = time.time() - total_start_time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")

    return history

# pcam_main.py
import time
import torch
import torch.nn as nn
import torch.optim as optim

# --- Your modules (adjust imports to match your folder names) ---
from datasets.pcam_dataset import get_pcam_loaders          # returns trainloader, valloader
from train.pcam_train import train_pcam                     # the training function I wrote for you

# If you have a PCAM CNN model file already:
from models.pcam_model import PCAMCNN                       # CNN for PCAM (expects N,C,H,W)

# If you have a generic MLPNet (or make one here)
class PCAMMLP(nn.Module):
    """Simple MLP for PCAM that takes flattened images."""
    def __init__(self, input_dim, num_classes=2, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def main():
    # ====== CONFIG (edit these) ======
    CONFIG = {
        "epochs": 11,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "num_classes": 2,

        # choose: "cnn" or "mlp"
        "model_type": "cnn",

        # PCAM H5 file paths (update if needed)
        "pcam_x_path": "camelyonpatch_level_2_split_train_x.h5",
        "pcam_y_path": "camelyonpatch_level_2_split_train_y.h5",
        "val_size": 0.2,
        "seed": 42,
    }

    total_start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Config:", CONFIG)

    # ====== DATA ======
    trainloader, valloader = get_pcam_loaders(
        batch_size=CONFIG["batch_size"],
        x_path=CONFIG["pcam_x_path"],
        y_path=CONFIG["pcam_y_path"],
        seed=CONFIG["seed"],
        val_size=CONFIG["val_size"],
    )

    # ====== MODEL ======
    model_type = CONFIG["model_type"].lower()
    if model_type == "cnn":
        model = PCAMCNN(num_classes=CONFIG["num_classes"]).to(device)
        is_mlp = False

    elif model_type == "mlp":
        # infer input_dim from one batch
        sample_images, _ = next(iter(trainloader))
        input_dim = sample_images.view(sample_images.size(0), -1).shape[1]
        model = PCAMMLP(input_dim=input_dim, num_classes=CONFIG["num_classes"]).to(device)
        is_mlp = True

    else:
        raise ValueError("CONFIG['model_type'] must be 'cnn' or 'mlp'.")

    # ====== LOSS + OPTIMIZER ======
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # ====== TRAIN (tracks loss/acc/f1/time per epoch + total training time) ======
    history = train_pcam(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=CONFIG["epochs"],
        is_mlp=is_mlp
    )

    # ====== OPTIONAL: SAVE HISTORY (so you can plot later) ======
    # If you want to save, uncomment:
    # import json
    # with open("pcam_history.json", "w") as f:
    #     json.dump(history, f, indent=2)

    total_program_time = time.time() - total_start_time
    print(f"\nTotal Program Time: {total_program_time:.2f} seconds")


if __name__ == "__main__":
    main()

import torch.nn as nn


class AdultMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(config["input_dim"], config["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["hidden_dim"], config["num_classes"])
        )

    def forward(self, x):
        return self.model(x)

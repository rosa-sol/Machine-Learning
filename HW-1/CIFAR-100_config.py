import torch

# Global configuration settings
num_epochs = 11
batch_size = 64
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Generated config.py content.")
print(f"num_epochs: {num_epochs}")
print(f"batch_size: {batch_size}")
print(f"learning_rate: {learning_rate}")
print(f"device: {device}")

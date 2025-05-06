import torch
import os
import numpy as np
import random
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from src.models.improved_model import create_improved_model

# For reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("Loading dataset...")
# Load the MUTAG dataset
dataset = TUDataset(root='./data/', name='MUTAG')
node_feature_dim = dataset.num_node_features
max_nodes = max([data.num_nodes for data in dataset])
print(f"Dataset loaded. Node feature dim: {node_feature_dim}, Max nodes: {max_nodes}")

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, val_dataset = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)], generator=rng)

# Create a simple dataloader
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Create model
print("Creating model...")
model = create_improved_model(
    node_feature_dim=node_feature_dim,
    hidden_dim=8,
    latent_dim=5,
    max_nodes=max_nodes,
    encoder_type='mp'
)

# Get a batch of data
print("Getting a batch of data...")
batch = next(iter(train_loader))
print(f"Batch: {batch}")
print(f"x shape: {batch.x.shape}")
print(f"edge_index shape: {batch.edge_index.shape}")
print(f"batch shape: {batch.batch.shape}")

# Test the model
print("Testing model forward pass...")
try:
    with torch.no_grad():
        loss = model(batch.x, batch.edge_index, batch.batch)
    print(f"Forward pass successful, loss: {loss.item()}")
except Exception as e:
    print(f"Error in model forward pass: {str(e)}")
    import traceback
    traceback.print_exc()
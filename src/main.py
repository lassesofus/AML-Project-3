import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import random
from erdos_renyi import ErdosRenyiSampler

#  Device
device = 'cpu'

#  Load the MUTAG dataset
dataset = TUDataset(root='./data/', name='MUTAG').to(device)

# If needed, use the full dataset as training data (or split into train/test)
train_dataset = dataset

# Set seed for reproducibility
torch.manual_seed(0)
random.seed(0)

# Initialize the sampler once
ER_sampler = ErdosRenyiSampler(train_dataset)

for i in range(5):
    # Sample an Erdös–Rényi graph
    N, r, edge_index, adj = ER_sampler.sample()
    print(f"Sampled Erdös–Rényi graph with {N} nodes and density {r:.4f}.")
    print("Edge index:", edge_index)
    print("Adjacency matrix:\n", adj)

    # Visualize the sampled graph
    plt.figure(figsize=(8, 6))
    plt.title(f"Erdös–Rényi Graph with {N} Nodes and Density {r:.4f}")
    plt.scatter(edge_index[0], edge_index[1], s=10, c='blue', alpha=0.5)
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")
    plt.xlim(0, N)
    plt.ylim(0, N)
    plt.grid()
    plt.show()



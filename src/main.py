import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import random
from erdos_renyi import ErdosRenyiSampler
from utils import compute_graph_hash, compute_metrics, graph_to_nx, sample_and_evaluate
from tqdm import tqdm
import pdb

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

# for i in range(5):
#     # Sample an Erdös–Rényi graph
#     N, r, edge_index, adj = ER_sampler.sample()
#     print(f"Sampled Erdös–Rényi graph with {N} nodes and density {r:.4f}.")
#     print("Edge index:", edge_index)
#     print("Adjacency matrix:\n", adj)

#     # Visualize the sampled graph
#     plt.figure(figsize=(8, 6))
#     plt.title(f"Erdös–Rényi Graph with {N} Nodes and Density {r:.4f}")
#     plt.scatter(edge_index[0], edge_index[1], s=10, c='blue', alpha=0.5)
#     plt.xlabel("Node Index")
#     plt.ylabel("Node Index")
#     plt.xlim(0, N)
#     plt.ylim(0, N)
#     plt.grid()
#     plt.show()

# Uncomment to compute and save WL hashes for the training set
# Precompute WL hashes for graphs in the training set
training_hashes = set()
for data in tqdm(train_dataset):
    # Assumes each data object has attributes: num_nodes and edge_index
    pdb.set_trace()
    training_hash = compute_graph_hash(data.num_nodes, data.edge_index)
    training_hashes.add(training_hash)

# # Save the training hashes to a file
# # Only 139 unique hashes in the training set
# with open('training_hashes.txt', 'w') as f:
#     for h in training_hashes:
#         f.write(f"{h}\n")

# Load the training hashes from the file
with open('training_hashes.txt', 'r') as f:
    training_hashes = {line.strip() for line in f}
print(f"Loaded {len(training_hashes)} training hashes.")

# Evaluate baseline: Erdös–Rényi sampler
baseline_metrics = sample_and_evaluate(ER_sampler, training_hashes=training_hashes, num_samples=1000)
print("Baseline Metrics:")
print("Novel: {:.2f}%".format(baseline_metrics[0]))
print("Unique: {:.2f}%".format(baseline_metrics[1]))
print("Novel and Unique: {:.2f}%".format(baseline_metrics[2]))

# Baseline Metrics:
# Novel: 100.00%
# Unique: 100.00%
# Novel and Unique: 100.00%

# Something is wrong with the sampling process, as all sampled graphs are novel and unique.










from torch_geometric.datasets import TUDataset
import random
import torch
from utils import compute_graph_hash, graph_to_nx
from tqdm import tqdm

#  Device
device = 'cpu'

#  Load the MUTAG dataset
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
# Convert to NetworkX graphs
empirical_graphs = [graph_to_nx(data.num_nodes, data.edge_index) for data in dataset]

# Set seed for reproducibility
torch.manual_seed(0)
random.seed(0)

#Precompute WL hashes for graphs in the training set
training_hashes = set()
for data in tqdm(dataset):
    # Assumes each data object has attributes: num_nodes and edge_index
    training_hash = compute_graph_hash(data.num_nodes, data.edge_index)
    training_hashes.add(training_hash)

# Save the training hashes to a file
# Only 139 unique hashes in the training set
with open('training_hashes.txt', 'w') as f:
    for h in training_hashes:
        f.write(f"{h}\n")
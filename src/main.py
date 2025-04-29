import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import random
from erdos_renyi import ErdosRenyiSampler
from utils import compute_metrics, graph_to_nx, get_graph_stats, plot_histograms
from tqdm import tqdm
import pdb
from networkx.algorithms import weisfeiler_lehman_graph_hash
import numpy as np

#  Device
device = 'cpu'

#  Load the MUTAG dataset
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
# Convert to NetworkX graphs
empirical_graphs = [graph_to_nx(data.num_nodes, data.edge_index) for data in dataset]

# Set seed for reproducibility
torch.manual_seed(0)
random.seed(0)

# Initialize the sampler once
ER_sampler = ErdosRenyiSampler(dataset)

# Uncomment to compute and save WL hashes for the training set
# Precompute WL hashes for graphs in the training set
# training_hashes = set()
# for data in tqdm(dataset):
#     # Assumes each data object has attributes: num_nodes and edge_index
#     training_hash = compute_graph_hash(data.num_nodes, data.edge_index)
#     training_hashes.add(training_hash)

# # Save the training hashes to a file
# # Only 139 unique hashes in the training set
# with open('training_hashes.txt', 'w') as f:
#     for h in training_hashes:
#         f.write(f"{h}\n")

#Load the training hashes from the file
with open('training_hashes.txt', 'r') as f:
    training_hashes = {line.strip() for line in f}
print(f"Loaded {len(training_hashes)} training hashes.")

# Sample 1000 graphs and evaluate
baseline_graphs = ER_sampler.sample_graphs(num_samples=1000)
sampled_hashes = []
for data in tqdm(baseline_graphs):
    sampled_hash = weisfeiler_lehman_graph_hash(data)
    sampled_hashes.append(sampled_hash)

# Compute metrics
novel_percentage, unique_percentage, novel_unique_percentage = compute_metrics(sampled_hashes, training_hashes)
print(f"Novel: {novel_percentage:.2f}%")
print(f"Unique: {unique_percentage:.2f}%")
print(f"Novel and Unique: {novel_unique_percentage:.2f}%")

# Compute and print graph statistics
baseline_stats = get_graph_stats(baseline_graphs)
empirical_stats = get_graph_stats(empirical_graphs)

plot_histograms(baseline_stats, empirical_stats)























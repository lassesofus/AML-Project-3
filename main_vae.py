import pdb
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from architecture import get_vae
from train import train_vae
import os

# Ensure directories exist
os.makedirs('./models', exist_ok=True)
os.makedirs('./epochs', exist_ok=True)

# Configs
device = 'cpu'
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
batch_size = 64
save_path = './models/graph_vae.pt'
debug_mode = False  # Enable debugging output to analyze posterior collapse

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Compute empirical distribution of the number of nodes in the dataset
num_nodes_list = [data.num_nodes for data in dataset]

vae = get_vae(num_nodes_list)

# Train with anti-collapse measures
train_vae(vae, dataloader, lr=5e-4, epochs=400, save_path=save_path, 
          degree_penalty_weight=10, debug_mode=debug_mode,  # Reduced penalty weight
          patience=100, min_delta=0.0001,
          kl_annealing=True, min_kl_weight=0.2, max_kl_weight=1.0)  # Added free bits and beta-VAE
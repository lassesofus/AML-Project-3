import pdb
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from architecture import get_vae
from train import train_vae


# Configs
device = 'cuda'
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
batch_size = 32
save_path = './models/graph_vae.pt'

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Compute empirical distribution of the number of nodes in the dataset
num_nodes_list = [data.num_nodes for data in dataset]

vae = get_vae(num_nodes_list)

# Train and save best model
train_vae(vae, dataloader, lr=1e-3,epochs=100, save_path=save_path)
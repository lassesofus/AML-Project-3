import os
import pdb
from matplotlib import pyplot as plt
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from graph_vae import GraphVAE, GraphEncoder, GRANDecoder,SimpleMLPDecoder, degree_loss_from_batch
from tqdm import tqdm
import torch.nn.functional as F

# Configs
device = 'cuda'
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
node_feature_dim = dataset.num_node_features  # should be 7
latent_dim = 16
batch_size = 16
save_path = './models/graph_vae.pt'
alpha = 0
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Compute empirical distribution of the number of nodes in the dataset
num_nodes_list = [data.num_nodes for data in dataset]
unique_node_counts, counts = torch.unique(torch.tensor(num_nodes_list), return_counts=True)
node_count_distribution = counts.float() / counts.sum()  # Normalize to get probabilities
max_nodes = unique_node_counts.max().item()  # Maximum number of nodes in the dataset

def train_vae(model, dataloader, epochs=50, lr=1e-3, save_path='graph_vae.pt', loss_plot_path='loss_curve.png'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    loss_history = []
    with tqdm(range(1, epochs + 1), desc="Training Epochs") as pbar:
        for epoch in pbar:
            model.train()
            total_loss = 0

            for batch in dataloader:
                batch = batch.to(device)
                # Compute num_nodes_per_graph: count nodes per graph
                num_nodes_per_graph = torch.bincount(batch.batch)
                optimizer.zero_grad()
                out = model(batch, num_nodes_per_graph)
                deg_loss = degree_loss_from_batch(out["recon_data"], batch, max_nodes=vae.decoder.max_nodes)

                loss = out["loss"] +alpha * deg_loss
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs

            avg_loss = total_loss / len(dataloader.dataset)
            loss_history.append(avg_loss)
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'recon_loss': f'{out["loss"].item():.4f}', 'deg_loss': f'{deg_loss.item():.4f}'})

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': best_loss,
                }, save_path)
                pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Best Loss': f'{best_loss:.4f}'})

    # Plot loss curve
    plt.figure()
    plt.plot(range(1, epochs + 1), loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GraphVAE Training Loss Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"📉 Saved loss curve to {loss_plot_path}")

encoder = GraphEncoder(node_feature_dim,latent_dim,latent_dim).to(device)
decoder = SimpleMLPDecoder(latent_dim,latent_dim,max_nodes).to(device)

vae = GraphVAE(encoder, decoder, latent_dim)

# Train and save best model
train_vae(vae, dataloader, epochs=100, save_path=save_path)
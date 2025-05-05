import torch
import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import negative_sampling, to_undirected, to_dense_adj

import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import scipy.optimize

from utils import graph_to_nx, plot_graphs

# Set seed for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class GraphVAE(torch.nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=64, latent_dim=32, loss_type="KL"):
        super(GraphVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = 30  # Maximum nodes in MUTAG dataset
        self.loss_type = loss_type
        
        # Encoder layers
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.mu_proj = torch.nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = torch.nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        self.latent_to_nodes = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim * self.max_nodes)
        )
        self.edge_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
        
        # Register loss functions
        self.loss_functions = {
            "KL": self._kl_loss,
            "WS": self._wasserstein_loss
        }
    
    def encode(self, x, edge_index, batch):
        # Node-level encoding
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        
        # Graph-level pooling
        pooled = global_mean_pool(h, batch)
        
        # Projection to latent parameters
        mu = self.mu_proj(pooled)
        logvar = self.logvar_proj(pooled)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z, num_nodes):
        # Generate node embeddings from latent
        batch_size = z.size(0)
        node_embeddings = self.latent_to_nodes(z)
        node_embeddings = node_embeddings.view(batch_size, self.max_nodes, self.hidden_dim)
        
        # Get single graph's embeddings
        n = num_nodes
        embeddings = node_embeddings[0, :n]
        
        # Create adjacency matrix
        adj_matrix = torch.zeros(n, n, device=z.device)
        
        # For each pair of nodes, predict edge probability
        for i in range(n):
            for j in range(i+1, n):  # Upper triangular
                edge_input = torch.cat([embeddings[i], embeddings[j]], dim=0)
                edge_prob = torch.sigmoid(self.edge_proj(edge_input))
                adj_matrix[i, j] = edge_prob
        
        # Make symmetric
        adj_matrix = adj_matrix + adj_matrix.t()
        
        return adj_matrix
    
    def forward(self, x, edge_index, batch, num_nodes):
        mu, logvar = self.encode(x, edge_index, batch)
        z = self.reparameterize(mu, logvar)
        adj_pred = self.decode(z, num_nodes)
        return adj_pred, mu, logvar
    
    def _kl_loss(self, mu, logvar, **kwargs):
        """KL divergence loss implementation"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    def _wasserstein_loss(self, mu, logvar, **kwargs):
        """Wasserstein distance loss implementation"""
        # Convert logvar to std
        std = torch.exp(0.5 * logvar)
        
        # Mean component: ||μ||²
        mean_term = torch.sum(mu.pow(2))
        
        # Covariance component: Σᵢ(σᵢ-1)²
        cov_term = torch.sum((std - 1).pow(2))
        
        # Total 2-Wasserstein distance squared
        w2_squared = mean_term + cov_term
        
        return w2_squared
    
    def compute_reg_loss(self, mu, logvar, **kwargs):
        """Compute regularization loss based on selected loss type"""
        if self.loss_type not in self.loss_functions:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Available types: {list(self.loss_functions.keys())}")
        return self.loss_functions[self.loss_type](mu, logvar, **kwargs)

    def kl_loss(self, mu, logvar):
        return self._kl_loss(mu, logvar)
    
    def wasserstein_loss(self, mu, logvar):
        return self._wasserstein_loss(mu, logvar)
    
    def hungarian_recon_loss(self, adj_pred, edge_index, num_nodes):
        # Convert edge_index to dense adjacency matrix
        batch = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        true_adj = to_dense_adj(edge_index, batch, max_num_nodes=num_nodes)[0]
        
        # Compute cost matrix for Hungarian algorithm
        cost_matrix = torch.cdist(adj_pred, true_adj, p=2)
        
        # Use Hungarian algorithm
        indices = scipy.optimize.linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        row_idx, col_idx = indices
        
        # Permute the predicted matrix
        adj_pred_permuted = adj_pred[row_idx]
        
        # Compute BCE loss
        loss = torch.nn.functional.binary_cross_entropy(
            adj_pred_permuted.view(-1), 
            true_adj.view(-1),
            reduction='mean'
        )
        return loss

def beta_schedule(epoch, warmup=50, beta_max=1.0):
    return min(1.0, epoch / warmup) * beta_max

def train_model(model, dataset, device, epochs=50, lr=0.001, warmup=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_reg_loss = 0  # Either KL or Wasserstein
        
        # Process each graph individually
        for i, data in enumerate(dataset):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Create batch index (single graph)
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            
            # Forward pass
            adj_pred, mu, logvar = model(data.x, data.edge_index, batch, data.num_nodes)
            
            # Compute losses
            recon_loss = model.hungarian_recon_loss(adj_pred, data.edge_index, data.num_nodes)
            reg_loss = model.compute_reg_loss(mu, logvar)
            
            # Total loss
            beta = beta_schedule(epoch, warmup=50)
            loss = recon_loss + beta * reg_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_reg_loss += reg_loss.item()
            
        # Average loss
        avg_loss = epoch_loss / len(dataset)
        avg_recon_loss = epoch_recon_loss / len(dataset)
        avg_reg_loss = epoch_reg_loss / len(dataset)
        history.append(avg_loss)
        
        # Log progress with individual loss components
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} | Total Loss: {avg_loss:.4f} | Recon: {avg_recon_loss:.4f} | {model.loss_type}: {avg_reg_loss:.4f} | Beta: {beta:.2f}")
    
    return history


def sample_graphs(model, num_samples, device, node_sampler):
    model.eval()
    samples = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Sample from latent space
            z = torch.randn(1, model.latent_dim).to(device)
            
            # Sample number of nodes
            num_nodes = node_sampler()
            
            # Generate adjacency matrix
            adj_pred = model.decode(z, num_nodes)
            
            # Threshold to binary
            adj_binary = (adj_pred > 0.5).cpu().numpy()
            
            # Create networkx graph
            G = nx.from_numpy_array(adj_binary)
            samples.append(G)
    return samples


def main():
    ### ------------------------- ###
    parser = argparse.ArgumentParser(description='Graph VAE')
    parser.add_argument('--mode', type=str, choices=['train', 'sample'], default='train',
                        help='Mode: train the model or sample graphs')
    parser.add_argument('--output_dir', type=str, default='./experiments/simple_graph_vae',
                        help='Directory to save results')
    parser.add_argument('--loss_type', type=str, default='KL', choices=['KL', 'WS'],
                        help='Type of regularization loss: KL divergence (KL) or Wasserstein (WS)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of graphs to sample')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    # Create a meaningful output directory name
    output_dir = Path(f"./experiments/graph_vae_{args.loss_type.lower()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare output directories
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
        
    # Load MUTAG dataset
    dataset = TUDataset(root='./data/', name='MUTAG')
    for data in dataset:
        data.to(device)
    
    # Convert to NetworkX graphs for visualization (empirical graphs)
    empirical_graphs = [graph_to_nx(data.num_nodes, data.edge_index) for data in dataset]
    
    # Create empirical node distribution sampler
    def empirical_node_sampler():
        return random.choice([data.num_nodes for data in dataset])
    
    
    # Model definition
    model = GraphVAE(
        node_feat_dim=dataset.num_features,
        hidden_dim=64,
        latent_dim=32,
        loss_type=args.loss_type
    ).to(device)
    
    # Model file
    model_file = output_dir / "model.pt"
    
    if args.mode == 'train':
        # Train model
        print(f"Training model with {args.loss_type} loss...")
        history = train_model(
            model=model,
            dataset=dataset,
            device=device,
            epochs=50,
            lr=0.001, 
            warmup=50
        )
        print("Training completed.")
        torch.save({
            'model_state_dict': model.state_dict(),
            'loss_type': args.loss_type,
        }, model_file)
        print(f"Model saved to {model_file}")
        
        # Plot loss history
        plt.figure()
        plt.plot(range(1, len(history) + 1), history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Graph VAE Training Loss ({args.loss_type}")
        plt.tight_layout()
        plt.savefig(fig_dir / "loss.png")
        
        # Sample graphs
        sampled_graphs = sample_graphs(
            model=model,
            num_samples=args.num_samples,
            device=device,
            node_sampler=empirical_node_sampler
        )
        
        # Plot sampled graphs
        plot_graphs(sampled_graphs, fig_dir, title=f'Generated Graphs ({args.loss_type})')
    
    elif args.mode == 'sample':
        # Load model if exists
        if model_file.exists():
            checkpoint = torch.load(model_file)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # New format with metadata
                model.load_state_dict(checkpoint['model_state_dict'])
                model.loss_type = checkpoint.get('loss_type', args.loss_type)
                print(f"Loaded model from {model_file} (Loss type: {model.loss_type})")
            else:
                # Legacy format without metadata
                model.load_state_dict(checkpoint)
                print(f"Loaded model from {model_file} (Loss type: {args.loss_type}, from args)")
        else:
            print(f"No model found at {model_file}. Please train model first.")
            return
        
        # Sample graphs
        sampled_graphs = sample_graphs(
            model=model,
            num_samples=args.num_samples,
            device=device,
            node_sampler=empirical_node_sampler
        )
        
        # Plot sampled graphs
        plot_graphs(sampled_graphs, fig_dir, title=f'Generated Graphs ({model.loss_type})')
        
        print(f"Generated {args.num_samples} graphs. Saved to {fig_dir}")


if __name__ == "__main__":
    main()
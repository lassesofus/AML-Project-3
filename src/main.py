import torch
import os
import numpy as np
import random

from src.models.model import GraphVAE
from src.utils.data import load_data, create_dataloaders
from src.training.train import train_graph_vae

# For reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Set random seed for reproducibility
    set_seed()
    
    # Create directories for outputs
    os.makedirs('figures', exist_ok=True)
    
    # Calculate target edge density from the dataset
    dataset, node_feature_dim, max_nodes, target_degree_dist, node_counts = load_data()
    total_edges = 0
    total_possible_edges = 0
    
    for data in dataset:
        n_edges = data.edge_index.shape[1] // 2  # Each edge counted twice
        n_nodes = data.num_nodes
        total_edges += n_edges
        total_possible_edges += (n_nodes * (n_nodes - 1)) // 2
    
    target_edge_density = total_edges / total_possible_edges
    print(f"Target edge density: {target_edge_density:.4f}")
    
    # Configuration
    config = {
        'hidden_dim': 64,
        'latent_dim': 32,
        'learning_rate': 0.0005,
        'num_epochs': 400,
        'batch_size': 32,
        'num_layers': 3,
        'model_save_path': 'graph_vae_model.pt',
        'max_nodes': max_nodes,
        'node_feature_dim': node_feature_dim,
        
        # KL annealing
        'kl_annealing_start': 0,
        'kl_annealing_end': 100,
        'final_beta': 0.1,
        
        # Structure penalty weights
        'distribution_weight': 200.0,
        'valency_weight': 50.0,
        'sparsity_weight': 100.0,
        'connectivity_weight': 100.0,
        
        # Structure constraints
        'max_degree': 3,
        'target_edge_density': target_edge_density,
        'target_degree_dist': target_degree_dist,
    }
    
    # Create data loaders
    train_loader, validation_loader, test_loader = create_dataloaders(
        dataset, 
        batch_size=config['batch_size']
    )
    
    # Initialize model
    model = GraphVAE(
        node_feature_dim, 
        config['hidden_dim'], 
        config['latent_dim'], 
        max_nodes, 
        config['num_layers'], 
        target_degree_dist
    ).to(device)
    
    # Train the model
    trained_model, train_history, val_history = train_graph_vae(
        model,
        train_loader,
        validation_loader,
        config
    )
    
    print(f"Training completed! Model saved to {config['model_save_path']}")
    print("Run 'evaluate.py' to evaluate the model performance.")

if __name__ == "__main__":
    main()
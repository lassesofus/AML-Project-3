import torch
import os
import numpy as np
import random
import argparse

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

def create_training_config(args=None):
    """
    Create training configuration with command line arguments
    
    Args:
        args: Command line arguments
    
    Returns:
        config: Dictionary with training configuration
    """
    # Load dataset info
    dataset, node_feature_dim, max_nodes, target_degree_dist, node_counts = load_data()
    
    # Default configuration - enhanced for connectivity-focused training
    config = {
        'hidden_dim': 64,      # Consistent with model dimensions
        'latent_dim': 32,      # Consistent with model dimensions
        'learning_rate': 0.0005, # Lower learning rate for stability
        'num_epochs': 100,     # Reasonable number of epochs
        'batch_size': 32,
        'num_layers': 3,
        'model_save_path': 'graph_vae_model.pt',
        'max_nodes': max_nodes,
        'node_feature_dim': node_feature_dim,
        
        # KL annealing
        'kl_annealing_start': 0,
        'kl_annealing_end': 50, # Shorter annealing period
        'final_beta': 0.1,     # Standard beta value for VAE
        
        # Connectivity parameters - using balanced weights
        'connectivity_weight': 0.5,  # Balanced weight for connectivity loss
        'degree_weight': 0.6,        # Slightly higher weight for degree distribution
        'early_stopping_patience': 30, # Patience for early stopping
        
        # Target degree distribution (for reference only)
        'target_degree_dist': target_degree_dist,
    }
    
    # Update configuration with command line arguments if provided
    if args:
        # Update values that are explicitly set via command line
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
    
    return config, dataset

def parse_arguments():
    """
    Parse command line arguments for training configuration
    
    Returns:
        args: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Train Graph VAE with enhanced connectivity focus')
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension size')
    parser.add_argument('--latent_dim', type=int, help='Latent dimension size')
    parser.add_argument('--num_layers', type=int, help='Number of message passing layers')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    
    # KL annealing
    parser.add_argument('--kl_annealing_start', type=int, help='Start epoch for KL annealing')
    parser.add_argument('--kl_annealing_end', type=int, help='End epoch for KL annealing')
    parser.add_argument('--final_beta', type=float, help='Final KL weight after annealing')
    
    # Connectivity parameters
    parser.add_argument('--connectivity_weight', type=float, help='Weight for connectivity loss')
    parser.add_argument('--degree_weight', type=float, help='Weight for degree distribution loss')
    parser.add_argument('--early_stopping_patience', type=int, help='Patience for early stopping')
    
    # Output
    parser.add_argument('--model_save_path', type=str, help='Path to save trained model')
    
    args = parser.parse_args()
    return args

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set random seed for reproducibility
    set_seed()
    
    # Create directories for outputs
    os.makedirs('figures', exist_ok=True)
    
    # Create training configuration
    config, dataset = create_training_config(args)
    
    # Print configuration
    print("\nTraining Configuration:")
    print(f"  Hidden Dimension: {config['hidden_dim']}")
    print(f"  Latent Dimension: {config['latent_dim']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Number of Epochs: {config['num_epochs']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  KL Annealing: {config['kl_annealing_start']} → {config['kl_annealing_end']} (beta={config['final_beta']})")
    print(f"  Connectivity Weight: {config['connectivity_weight']}")
    print(f"  Degree Weight: {config['degree_weight']}")
    print(f"  Early Stopping Patience: {config['early_stopping_patience']}")
    
    # Create data loaders
    train_loader, validation_loader, test_loader = create_dataloaders(
        dataset, 
        batch_size=config['batch_size']
    )
    
    # Initialize model
    model = GraphVAE(
        config['node_feature_dim'], 
        config['hidden_dim'], 
        config['latent_dim'], 
        config['max_nodes'], 
        config['num_layers']
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
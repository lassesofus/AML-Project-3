import torch
import os
import numpy as np
import random
import argparse
import traceback
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm

from src.models.improved_model import create_improved_model
from src.training.improved_train import train_improved_model, sample_graph_sizes

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

def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Improved Graph VAE for molecular graphs')
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample', 'eval'],
                            help='Mode (train, sample, or eval)')
        parser.add_argument('--encoder', type=str, default='mp', choices=['mp', 'conv'],
                            help='Encoder type (mp for message passing, conv for convolution)')
        parser.add_argument('--hidden_dim', type=int, default=8,
                            help='Hidden dimension size')
        parser.add_argument('--latent_dim', type=int, default=5,
                            help='Latent dimension size')
        parser.add_argument('--learning_rate', type=float, default=1e-4,
                            help='Learning rate')
        parser.add_argument('--batch_size', type=int, default=64,
                            help='Batch size')
        parser.add_argument('--num_epochs', type=int, default=1000,
                            help='Number of training epochs')
        parser.add_argument('--early_stopping_patience', type=int, default=100,
                            help='Patience for early stopping')
        parser.add_argument('--model_path', type=str, default='graph_vae_model_improved.pt',
                            help='Path to save/load the model')
        parser.add_argument('--samples', type=int, default=10,
                            help='Number of graphs to sample')
        parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                            help='Device to use (cpu, cuda, mps)')
        args = parser.parse_args()
        
        # Set random seed
        set_seed()
        
        # Create output directories
        os.makedirs('figures', exist_ok=True)
        
        # Determine device
        device = torch.device(args.device)
        
        print("Loading dataset...")
        # Load the MUTAG dataset
        dataset = TUDataset(root='./data/', name='MUTAG')
        node_feature_dim = dataset.num_node_features
        max_nodes = max([data.num_nodes for data in dataset])
        print(f"Dataset loaded. Node feature dim: {node_feature_dim}, Max nodes: {max_nodes}")
        
        # Split into training, validation, and test sets
        rng = torch.Generator().manual_seed(0)
        train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        # Create configuration
        config = {
            'hidden_dim': args.hidden_dim,
            'latent_dim': args.latent_dim,
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs,
            'early_stopping_patience': args.early_stopping_patience,
            'model_save_path': args.model_path,
            'max_nodes': max_nodes,
        }
        
        # Create graph size sampling function
        size_fn = sample_graph_sizes(dataset)
        
        # Mode handling
        if args.mode == 'train':
            print("\nTraining Configuration:")
            print(f"  Encoder: {args.encoder}")
            print(f"  Hidden Dimension: {config['hidden_dim']}")
            print(f"  Latent Dimension: {config['latent_dim']}")
            print(f"  Learning Rate: {config['learning_rate']}")
            print(f"  Number of Epochs: {config['num_epochs']}")
            print(f"  Batch Size: {args.batch_size}")
            print(f"  Early Stopping Patience: {config['early_stopping_patience']}")
            print(f"  Max Nodes: {config['max_nodes']}")
            print(f"  Device: {args.device}")
            
            # Create model
            print("Creating model...")
            model = create_improved_model(
                node_feature_dim=node_feature_dim,
                hidden_dim=config['hidden_dim'],
                latent_dim=config['latent_dim'],
                max_nodes=config['max_nodes'],
                encoder_type=args.encoder
            ).to(device)
            
            # Set the graph size sampling function
            model.set_sample_size_function(size_fn)
            
            # Train model
            print("Starting training...")
            trained_model, train_losses, val_losses = train_improved_model(
                model, train_loader, validation_loader, config
            )
            
            print(f"\nTraining completed! Model saved to {config['model_save_path']}")
        
        elif args.mode == 'sample':
            print(f"\nSampling {args.samples} graphs using model from {args.model_path}")
            
            # Create model
            model = create_improved_model(
                node_feature_dim=node_feature_dim,
                hidden_dim=config['hidden_dim'],
                latent_dim=config['latent_dim'],
                max_nodes=config['max_nodes'],
                encoder_type=args.encoder
            ).to(device)
            
            # Set the graph size sampling function
            model.set_sample_size_function(size_fn)
            
            # Load model weights
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model.eval()
            
            # Sample graphs
            with torch.no_grad():
                sampled_adjs = model.sample(args.samples)
                
            # Show basic statistics of sampled graphs
            print("\nSampled Graph Statistics:")
            
            # Count number of nodes in each graph (as sum of nodes with at least one edge)
            non_zeros = (sampled_adjs.sum(dim=1) > 0).float().sum(dim=1)
            print(f"  Average Nodes: {non_zeros.mean().item():.2f}")
            print(f"  Min Nodes: {non_zeros.min().item():.0f}")
            print(f"  Max Nodes: {non_zeros.max().item():.0f}")
            
            # Count average degree
            degrees = sampled_adjs.sum(dim=1).sum(dim=1) / non_zeros
            print(f"  Average Degree: {degrees.mean().item():.2f}")
            
            # Count disconnected graphs
            disconnected = 0
            for i in range(sampled_adjs.size(0)):
                # Get the adjacency matrix for this graph
                adj = sampled_adjs[i]
                # Get number of nodes
                n_nodes = int(non_zeros[i].item())
                if n_nodes <= 1:
                    disconnected += 1
                    continue
                
                # Extract the subgraph with actual nodes
                subgraph = adj[:n_nodes, :n_nodes]
                
                # Check if the graph is disconnected
                # This is a simple check that works for small graphs
                # More efficient algorithms exist for large graphs
                visited = set()
                def dfs(node):
                    visited.add(node)
                    for j in range(n_nodes):
                        if subgraph[node, j] > 0.5 and j not in visited:
                            dfs(j)
                
                # Start DFS from node 0
                dfs(0)
                if len(visited) < n_nodes:
                    disconnected += 1
            
            print(f"  Disconnected Graphs: {disconnected}/{args.samples} ({disconnected/args.samples*100:.1f}%)")
            
        elif args.mode == 'eval':
            print("Evaluation mode not yet implemented")
            # TODO: Implement evaluation mode similar to the original implementation
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        
if __name__ == "__main__":
    main()
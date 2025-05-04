import torch
import matplotlib.pyplot as plt
import numpy as np
from src.utils.data import prepare_batch_targets
from src.utils.plot import plot_curves
from src.training.loss import calculate_vae_loss, kl_annealing_factor
from src.utils.graph_utils import count_connected_components

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model, loader, max_nodes, node_feature_dim, beta, target_degree_dist, config, device):
    """
    Evaluate the model on a data loader
    
    Args:
        model: GraphVAE model
        loader: DataLoader for evaluation
        max_nodes: Maximum number of nodes per graph
        node_feature_dim: Dimension of node features
        beta: KL divergence weight
        target_degree_dist: Target degree distribution [7]
        config: Configuration dictionary
        device: Device to run evaluation on
        
    Returns:
        results: Dictionary of evaluation metrics
    """
    model.eval()
    total_results = {
        'total': 0,
        'reconstruction': 0,
        'kl': 0,
        'structure_penalty': 0,
        'degree_dist': 0,
        'valency': 0,
        'sparsity': 0,
        'connectivity': 0,
        'disconnected_graphs': 0
    }
    total_batches = 0
    total_graphs = 0
    pred_degree_dist_sum = torch.zeros_like(target_degree_dist)
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Prepare batch data
            adj_target, node_target, node_mask, _ = prepare_batch_targets(
                data, max_nodes, node_feature_dim, device
            )
            batch_size = adj_target.size(0)
            total_graphs += batch_size
            
            # Forward pass
            node_features, adj_logits, z_mean, z_logvar = model(
                data.x, data.edge_index, data.batch
            )
            
            # Compute loss components
            loss_dict = calculate_vae_loss(
                adj_logits, node_features, adj_target, node_target, node_mask,
                z_mean, z_logvar, beta, target_degree_dist, config
            )
            
            # Update metrics
            for k in total_results.keys():
                if k in loss_dict:
                    total_results[k] += loss_dict[k].item() * batch_size
            
            # Sum predicted degree distributions
            pred_degree_dist_sum += loss_dict['pred_degree_dist'] * batch_size
            
            # Count disconnected graphs
            adj_probs = torch.sigmoid(adj_logits)
            for b in range(batch_size):
                graph_adj = adj_probs[b]
                graph_mask = node_mask[b]
                if graph_mask.sum() > 1:  # Skip empty graphs
                    n_components = count_connected_components(graph_adj, graph_mask)
                    if n_components > 1:
                        total_results['disconnected_graphs'] += 1
            
            total_batches += 1
    
    # Calculate averages
    results = {k: v / total_graphs for k, v in total_results.items()}
    results['disconnected_ratio'] = total_results['disconnected_graphs'] / total_graphs
    results['pred_degree_dist'] = pred_degree_dist_sum / total_graphs
    results['target_degree_dist'] = target_degree_dist
    
    return results

def train_graph_vae(model, train_loader, validation_loader, config):
    """
    Train the Graph VAE model with the specified configuration
    
    Args:
        model: GraphVAE model
        train_loader: DataLoader for training
        validation_loader: DataLoader for validation
        config: Configuration dictionary
    
    Returns:
        model: Trained model
        train_history: Training metrics history
        val_history: Validation metrics history
    """
    # Extract configuration parameters
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    max_nodes = config['max_nodes']
    node_feature_dim = config['node_feature_dim']
    target_degree_dist = config['target_degree_dist'].to(device)
    model_save_path = config['model_save_path']
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize history trackers
    train_history = {
        'total': [], 'reconstruction': [], 'kl': [], 'structure_penalty': [],
        'degree_dist': [], 'valency': [], 'sparsity': [], 'connectivity': [],
        'pred_degree_dist': [], 'target_degree_dist': []
    }
    
    val_history = {
        'total': [], 'reconstruction': [], 'kl': [], 'structure_penalty': [],
        'degree_dist': [], 'valency': [], 'sparsity': [], 'connectivity': [],
        'disconnected_ratio': [], 'pred_degree_dist': [], 'target_degree_dist': []
    }
    
    # Setup plotting
    plt.ion()
    
    # Training Loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {
            'total': 0.0, 'reconstruction': 0.0, 'kl': 0.0, 'structure_penalty': 0.0,
            'degree_dist': 0.0, 'valency': 0.0, 'sparsity': 0.0, 'connectivity': 0.0
        }
        pred_degree_dist_sum = torch.zeros_like(target_degree_dist)
        num_batches = 0
        
        # Calculate current KL annealing factor
        current_beta = kl_annealing_factor(epoch, config)
        
        for data in train_loader:
            data = data.to(device)
            batch_size = data.batch.max().item() + 1
            
            # Prepare batch data
            adj_target, node_target, node_mask, _ = prepare_batch_targets(
                data, max_nodes, node_feature_dim, device
            )
            
            # Forward pass
            node_features, adj_logits, z_mean, z_logvar = model(
                data.x, data.edge_index, data.batch
            )
            
            # Compute loss
            loss_dict = calculate_vae_loss(
                adj_logits, node_features, adj_target, node_target, node_mask,
                z_mean, z_logvar, current_beta, target_degree_dist, config
            )
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()
            
            # Update metrics (only scalar values)
            for k in epoch_losses.keys():
                epoch_losses[k] += loss_dict[k].item()
            
            # Handle tensor metrics separately
            pred_degree_dist_sum += loss_dict['pred_degree_dist']
            
            num_batches += 1
        
        # Record training metrics
        for k, v in epoch_losses.items():
            train_history[k].append(v / num_batches)
        
        # Record tensor metrics
        train_history['pred_degree_dist'].append(pred_degree_dist_sum / num_batches)
        train_history['target_degree_dist'].append(target_degree_dist)
        
        # Evaluate on validation set
        val_results = evaluate_model(
            model, validation_loader, max_nodes, node_feature_dim,
            current_beta, target_degree_dist, config, device
        )
        
        for k, v in val_results.items():
            if k in val_history:
                val_history[k].append(v)
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"--- Epoch {epoch+1}/{num_epochs} --- Beta: {current_beta:.4f}")
            print(f"  Train | Loss: {train_history['total'][-1]:.4f} | "
                  f"Recon: {train_history['reconstruction'][-1]:.4f} | "
                  f"KL: {train_history['kl'][-1]:.4f}")
            print(f"        | Degree Dist: {train_history['degree_dist'][-1]:.4f} | "
                  f"Valency: {train_history['valency'][-1]:.4f} | "
                  f"Sparsity: {train_history['sparsity'][-1]:.4f} | "
                  f"Conn: {train_history['connectivity'][-1]:.4f}")
            
            print(f"  Valid | Loss: {val_history['total'][-1]:.4f} | "
                  f"Recon: {val_history['reconstruction'][-1]:.4f} | "
                  f"KL: {val_history['kl'][-1]:.4f}")
            print(f"        | Degree Dist: {val_history['degree_dist'][-1]:.4f} | "
                  f"Valency: {val_history['valency'][-1]:.4f} | "
                  f"Sparsity: {val_history['sparsity'][-1]:.4f} | "
                  f"Conn: {val_history['connectivity'][-1]:.4f} | "
                  f"Disconn: {val_history['disconnected_ratio'][-1]:.2%}")
            
            # Update plots
            plot_curves(train_history, val_history, epoch)
    
    # Save Model
    print(f"Training finished. Saving model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)
    plt.ioff()
    plt.close('all')
    
    return model, train_history, val_history
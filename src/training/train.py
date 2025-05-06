import torch
import matplotlib.pyplot as plt
import os
import networkx as nx
from src.utils.data import prepare_batch_targets
from src.training.loss import calculate_vae_loss, kl_annealing_factor

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_loss_plot(train_history, val_history, epoch, save_dir='figures'):
    """
    Save a plot of the training and validation loss
    
    Args:
        train_history: Dictionary of training metrics
        val_history: Dictionary of validation metrics
        epoch: Current epoch number
        save_dir: Directory to save plot images
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure for loss curves
    plt.figure(figsize=(15, 8))
    
    # Plot total loss
    plt.subplot(2, 2, 1)
    plt.plot(train_history['total'], label='Train Total Loss')
    plt.plot(val_history['total'], label='Val Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Total Loss (Epoch {epoch})')
    
    # Plot reconstruction and KL loss
    plt.subplot(2, 2, 2)
    plt.plot(train_history['reconstruction'], label='Train Recon')
    plt.plot(val_history['reconstruction'], label='Val Recon')
    plt.plot(train_history['kl'], label='Train KL')
    plt.plot(val_history['kl'], label='Val KL')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Reconstruction and KL Loss')
    
    # Plot connectivity loss
    if 'connectivity' in train_history:
        plt.subplot(2, 2, 3)
        plt.plot(train_history['connectivity'], label='Train Connectivity')
        plt.plot(val_history['connectivity'], label='Val Connectivity')
        plt.xlabel('Epoch')
        plt.ylabel('Connectivity Loss')
        plt.legend()
        plt.title('Connectivity Loss (Lower = Better Connected)')
    
    # Plot degree distribution loss if available
    if 'degree' in train_history:
        plt.subplot(2, 2, 4)
        plt.plot(train_history['degree'], label='Train Degree Dist.')
        plt.plot(val_history['degree'], label='Val Degree Dist.')
        plt.xlabel('Epoch')
        plt.ylabel('Degree Distribution Loss')
        plt.legend()
        plt.title('Degree Distribution Loss')
    
    # Save the figure with a fixed filename
    plt.tight_layout()
    filename = f'{save_dir}/loss_plot.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Updated loss plot at {filename} (Epoch {epoch})")

def compute_target_degree_distribution(loader, max_degree=5):
    """
    Compute the target degree distribution from the training data
    
    Args:
        loader: DataLoader containing training graphs
        max_degree: Maximum degree to consider
    
    Returns:
        degree_dist: Dictionary mapping degrees to their probabilities
    """
    degree_counts = {i: 0 for i in range(max_degree + 1)}
    total_nodes = 0
    
    # Collect degree statistics
    for data in loader:
        for graph_idx in range(data.num_graphs):
            # Get subgraph for this graph
            node_mask = data.batch == graph_idx
            edge_mask = torch.isin(data.edge_index[0], torch.where(node_mask)[0])
            subgraph_edges = data.edge_index[:, edge_mask]
            
            # Count degrees
            degrees = torch.bincount(subgraph_edges[0])
            for degree, count in enumerate(degrees):
                if degree <= max_degree:
                    degree_counts[degree] += count.item()
                else:
                    degree_counts[max_degree] += count.item()
            
            total_nodes += node_mask.sum().item()
    
    # Convert to probability distribution
    degree_dist = {degree: count / total_nodes for degree, count in degree_counts.items()}
    return degree_dist

def evaluate_model(model, loader, max_nodes, node_feature_dim, beta, device, target_degree_dist=None, config=None):
    """
    Evaluate the model on a data loader
    
    Args:
        model: GraphVAE model
        loader: DataLoader for evaluation
        max_nodes: Maximum number of nodes per graph
        node_feature_dim: Dimension of node features
        beta: KL divergence weight
        device: Device to run evaluation on
        target_degree_dist: Target degree distribution
        config: Configuration dictionary
        
    Returns:
        results: Dictionary of evaluation metrics
    """
    model.eval()
    total_results = {
        'total': 0.0,
        'reconstruction': 0.0,
        'kl': 0.0,
        'connectivity': 0.0,
        'degree': 0.0
    }
    total_graphs = 0
    
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
            node_features, adj_logits, z_mean, z_logvar, _ = model(
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
    
    # Calculate averages
    results = {k: v / total_graphs for k, v in total_results.items() if v > 0}
    
    return results

def connectivity_evaluation(model, loader, max_nodes, device):
    """
    Evaluate the percentage of generated graphs that are connected
    
    Args:
        model: Trained GraphVAE model
        loader: DataLoader with graph data for node counts
        max_nodes: Maximum number of nodes per graph
        device: Device to run evaluation on
        
    Returns:
        connected_percentage: Percentage of connected generated graphs
    """
    model.eval()
    all_generated = 0
    connected_count = 0
    
    # Get realistic node counts from the dataset
    node_counts = []
    for data in loader:
        for graph_idx in range(data.num_graphs):
            node_mask = data.batch == graph_idx
            node_counts.append(node_mask.sum().item())
    
    # Generate graphs with these node counts
    with torch.no_grad():
        for i in range(0, len(node_counts), 10):  # Process in batches of 10
            batch_node_counts = node_counts[i:i+10]
            batch_size = len(batch_node_counts)
            
            # Generate graphs
            _, sampled_adj, node_mask = model.sample(batch_size, batch_node_counts)
            
            # Check connectivity of each graph
            for b in range(batch_size):
                all_generated += 1
                n_nodes = int(node_mask[b].sum().item())
                
                # Extract adjacency matrix for this graph
                adj_matrix = sampled_adj[b, :n_nodes, :n_nodes].cpu().numpy()
                
                # Use networkx to check connectivity
                G = nx.from_numpy_array(adj_matrix)
                if nx.is_connected(G):
                    connected_count += 1
    
    # Calculate percentage
    connected_percentage = connected_count / all_generated * 100 if all_generated > 0 else 0
    return connected_percentage

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
    model_save_path = config['model_save_path']
    
    # Compute target degree distribution from training data
    print("Computing target degree distribution from training data...")
    target_degree_dist = compute_target_degree_distribution(train_loader)
    print(f"Target degree distribution: {target_degree_dist}")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add scheduler for learning rate decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
    )
    
    # Initialize history trackers
    train_history = {
        'total': [],
        'reconstruction': [],
        'kl': [],
        'connectivity': [],
        'degree': []
    }
    
    val_history = {
        'total': [],
        'reconstruction': [],
        'kl': [],
        'connectivity': [],
        'degree': []
    }
    
    # Set up the configuration for loss computation
    loss_config = {
        'connectivity_weight': config.get('connectivity_weight', 1.0),
        'degree_weight': config.get('degree_weight', 0.5)
    }
    
    # Setup plotting
    plt.ion()
    
    # Training Loop
    print("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'kl': 0.0,
            'connectivity': 0.0,
            'degree': 0.0
        }
        num_batches = 0
        
        # Calculate current KL annealing factor
        current_beta = kl_annealing_factor(epoch, config)
        
        for data in train_loader:
            data = data.to(device)
            
            # Prepare batch data
            adj_target, node_target, node_mask, _ = prepare_batch_targets(
                data, max_nodes, node_feature_dim, device
            )
            
            # Forward pass
            node_features, adj_logits, z_mean, z_logvar, _ = model(
                data.x, data.edge_index, data.batch
            )
            
            # Compute loss with connectivity and degree distribution components
            loss_dict = calculate_vae_loss(
                adj_logits, node_features, adj_target, node_target, node_mask,
                z_mean, z_logvar, current_beta, target_degree_dist, loss_config
            )
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss_dict['total'].backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            for k in epoch_losses.keys():
                if k in loss_dict:
                    epoch_losses[k] += loss_dict[k].item()
            
            num_batches += 1
        
        # Record training metrics
        for k, v in epoch_losses.items():
            if num_batches > 0:
                train_history[k].append(v / num_batches)
            else:
                train_history[k].append(0.0)
        
        # Evaluate on validation set
        val_results = evaluate_model(
            model, validation_loader, max_nodes, node_feature_dim,
            current_beta, device, target_degree_dist, loss_config
        )
        
        for k in val_history.keys():
            val_history[k].append(val_results.get(k, 0.0))
        
        # Update learning rate scheduler
        scheduler.step(val_results['total'])
        
        # Early stopping logic
        if val_results['total'] < best_val_loss:
            best_val_loss = val_results['total']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), model_save_path.replace('.pt', '_best.pt'))
        else:
            patience_counter += 1
        
        # Log progress
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"--- Epoch {epoch+1}/{num_epochs} --- Beta: {current_beta:.4f} --- LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Train | Loss: {train_history['total'][-1]:.4f} | Recon: {train_history['reconstruction'][-1]:.4f} | KL: {train_history['kl'][-1]:.4f}")
            
            if 'connectivity' in train_history:
                print(f"         | Connect: {train_history['connectivity'][-1]:.4f} | Degree: {train_history['degree'][-1]:.4f}")
            
            print(f"  Valid | Loss: {val_history['total'][-1]:.4f} | Recon: {val_history['reconstruction'][-1]:.4f} | KL: {val_history['kl'][-1]:.4f}")
            
            if 'connectivity' in val_history:
                print(f"         | Connect: {val_history['connectivity'][-1]:.4f} | Degree: {val_history['degree'][-1]:.4f}")
            
            # Every 20 epochs, evaluate graph connectivity on a small sample
            if (epoch + 1) % 20 == 0:
                connected_pct = connectivity_evaluation(model, validation_loader, max_nodes, device)
                print(f"  Generated Graph Connectivity: {connected_pct:.2f}% connected")
        
        # Save loss plots periodically
        if (epoch + 1) % 20 == 0:
            save_loss_plot(train_history, val_history, epoch + 1)
        
        # Early stopping check
        if patience_counter >= config.get('early_stopping_patience', 50):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    best_model_path = model_save_path.replace('.pt', '_best.pt')
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
    
    # Save final model
    print(f"Training finished. Saving final model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)
    
    # Final connectivity evaluation
    final_connected_pct = connectivity_evaluation(model, validation_loader, max_nodes, device)
    print(f"Final Generated Graph Connectivity: {final_connected_pct:.2f}% connected")
    
    plt.ioff()
    plt.close('all')
    
    return model, train_history, val_history
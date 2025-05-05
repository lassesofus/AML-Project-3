import torch
import matplotlib.pyplot as plt
import os
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
    plt.figure(figsize=(12, 6))
    
    # Plot total loss
    plt.subplot(1, 2, 1)
    plt.plot(train_history['total'], label='Train Total Loss')
    plt.plot(val_history['total'], label='Val Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Total Loss (Epoch {epoch})')
    
    # Plot reconstruction and KL loss
    plt.subplot(1, 2, 2)
    plt.plot(train_history['reconstruction'], label='Train Recon')
    plt.plot(val_history['reconstruction'], label='Val Recon')
    plt.plot(train_history['kl'], label='Train KL')
    plt.plot(val_history['kl'], label='Val KL')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Reconstruction and KL Loss')
    
    # Save the figure with a fixed filename
    plt.tight_layout()
    filename = f'{save_dir}/loss_plot.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Updated loss plot at {filename} (Epoch {epoch})")

def evaluate_model(model, loader, max_nodes, node_feature_dim, beta, device):
    """
    Evaluate the model on a data loader
    
    Args:
        model: GraphVAE model
        loader: DataLoader for evaluation
        max_nodes: Maximum number of nodes per graph
        node_feature_dim: Dimension of node features
        beta: KL divergence weight
        device: Device to run evaluation on
        
    Returns:
        results: Dictionary of evaluation metrics
    """
    model.eval()
    total_results = {
        'total': 0.0,
        'reconstruction': 0.0,
        'kl': 0.0
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
            node_features, adj_logits, z_mean, z_logvar = model(
                data.x, data.edge_index, data.batch
            )
            
            # Compute loss components
            loss_dict = calculate_vae_loss(
                adj_logits, node_features, adj_target, node_target, node_mask,
                z_mean, z_logvar, beta
            )
            
            # Update metrics
            for k in total_results.keys():
                total_results[k] += loss_dict[k].item() * batch_size
    
    # Calculate averages
    results = {k: v / total_graphs for k, v in total_results.items()}
    
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
    model_save_path = config['model_save_path']
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize history trackers
    train_history = {
        'total': [],
        'reconstruction': [],
        'kl': [] 
    }
    
    val_history = {
        'total': [],
        'reconstruction': [],
        'kl': []
    }
    
    # Setup plotting
    plt.ion()
    
    # Training Loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'kl': 0.0
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
            node_features, adj_logits, z_mean, z_logvar = model(
                data.x, data.edge_index, data.batch
            )
            
            # Compute loss
            loss_dict = calculate_vae_loss(
                adj_logits, node_features, adj_target, node_target, node_mask,
                z_mean, z_logvar, current_beta
            )
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()
            
            # Update metrics
            for k in epoch_losses.keys():
                epoch_losses[k] += loss_dict[k].item()
            
            num_batches += 1
        
        # Record training metrics
        for k, v in epoch_losses.items():
            train_history[k].append(v / num_batches)
        
        # Evaluate on validation set
        val_results = evaluate_model(
            model, validation_loader, max_nodes, node_feature_dim,
            current_beta, device
        )
        
        for k, v in val_results.items():
            val_history[k].append(v)
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"--- Epoch {epoch+1}/{num_epochs} --- Beta: {current_beta:.4f}")
            print(f"  Train | Loss: {train_history['total'][-1]:.4f} | "
                  f"Recon: {train_history['reconstruction'][-1]:.4f} | "
                  f"KL: {train_history['kl'][-1]:.4f}")
            
            print(f"  Valid | Loss: {val_history['total'][-1]:.4f} | "
                  f"Recon: {val_history['reconstruction'][-1]:.4f} | "
                  f"KL: {val_history['kl'][-1]:.4f}")
        
        # Save loss plots every 30 epochs
        if (epoch + 1) % 30 == 0:
            save_loss_plot(train_history, val_history, epoch + 1)
    
    # Save Model
    print(f"Training finished. Saving model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)
    plt.ioff()
    plt.close('all')
    
    return model, train_history, val_history
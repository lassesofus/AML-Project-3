import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_improved_model(model, train_loader, validation_loader, config):
    """
    Train an improved Graph VAE model.

    Parameters:
    model: ImprovedGraphVAE
        The VAE model to train
    train_loader: DataLoader
        DataLoader for training data
    validation_loader: DataLoader
        DataLoader for validation data
    config: dict
        Training configuration

    Returns:
    model: ImprovedGraphVAE
        Trained model
    train_losses: list
        Training losses per epoch
    validation_losses: list
        Validation losses per epoch
    """
    device = next(model.parameters()).device
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    
    # Initialize lists to store losses
    train_losses = []
    validation_losses = []
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    model.train()
    tqdm_range = tqdm(range(config['num_epochs']))
    for epoch in tqdm_range:
        # Train for one epoch
        train_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            
            # Forward pass
            loss = model(data.x, data.edge_index, data.batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item() * data.num_graphs
        
        # Compute average loss for the epoch
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in validation_loader:
                data = data.to(device)
                loss = model(data.x, data.edge_index, data.batch)
                val_loss += loss.item() * data.num_graphs
        
        # Compute average validation loss
        val_loss /= len(validation_loader.dataset)
        validation_losses.append(val_loss)
        
        # Update progress bar
        tqdm_range.set_postfix({'train_loss': train_loss, 'val_loss': val_loss})
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), config['model_save_path'])
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Switch back to training mode
        model.train()
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figures/loss_plot.png')
    
    # Load the best model
    model.load_state_dict(torch.load(config['model_save_path']))
    
    return model, train_losses, validation_losses

def sample_graph_sizes(dataset):
    """
    Create a function that samples graph sizes from the dataset distribution.
    
    Parameters:
    dataset: list
        List of graph data objects
    
    Returns:
    sample_fn: function
        Function that samples graph sizes
    """
    # Extract node counts from all graphs in the dataset
    node_counts = []
    for data in dataset:
        node_counts.append(data.num_nodes)
    
    # Convert to numpy array for easier manipulation
    node_counts = np.array(node_counts)
    
    # Get unique sizes and their counts
    sizes, counts = np.unique(node_counts, return_counts=True)
    probabilities = counts / counts.sum()
    
    # Create sampling function
    def sample_fn(n_samples):
        return np.random.choice(sizes, p=probabilities, size=n_samples)
    
    return sample_fn
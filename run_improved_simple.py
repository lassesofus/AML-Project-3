import torch
import os
import numpy as np
import random
import argparse
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt

from src.models.improved_model import create_improved_model

# For reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create output directories
os.makedirs('figures', exist_ok=True)

print("Loading dataset...")
# Load the MUTAG dataset
dataset = TUDataset(root='./data/', name='MUTAG')
node_feature_dim = dataset.num_node_features
max_nodes = max([data.num_nodes for data in dataset])
print(f"Dataset loaded. Node feature dim: {node_feature_dim}, Max nodes: {max_nodes}")

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, val_dataset = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)], generator=rng)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Create model
print("Creating model...")
model = create_improved_model(
    node_feature_dim=node_feature_dim,
    hidden_dim=16,  # Increased hidden dimension
    latent_dim=5,
    max_nodes=max_nodes,
    num_message_passing_rounds=3,  # Reduced complexity
    encoder_type='mp'
).to(device)

# Set beta value for KL divergence weight
model.beta = 0.01  # Start with very small beta for reconstruction focus

# Extract node counts from all graphs in the dataset for sampling
node_counts = []
for data in dataset:
    node_counts.append(data.num_nodes)
node_counts = np.array(node_counts)
sizes, counts = np.unique(node_counts, return_counts=True)
probabilities = counts / counts.sum()

def sample_size_fn(n_samples):
    return np.random.choice(sizes, p=probabilities, size=n_samples)

model.set_sample_size_function(sample_size_fn)

# Setup optimizer with smaller learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

# Setup learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-5)

# Initialize lists to store losses
train_losses = []
validation_losses = []

# Early stopping parameters
best_val_loss = float('inf')
patience_counter = 0
patience = 30

# Training loop
model.train()
num_epochs = 150
print(f"Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    # Train for one epoch
    model.train()
    train_loss = 0.0
    batch_count = 0
    
    for data in train_loader:
        data = data.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # KL annealing - start with small beta and increase gradually
        if epoch < 50:
            model.beta = min(0.05, 0.001 * (epoch + 1))
        
        loss = model(data.x, data.edge_index, data.batch)
        
        # Check if loss is valid
        if not torch.isnan(loss) and not torch.isinf(loss):
            # Backward pass and optimize
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item()
            batch_count += 1
    
    # Skip epoch if all batches had invalid loss
    if batch_count == 0:
        print(f"Epoch {epoch+1}/{num_epochs} - Skipping due to invalid losses")
        continue
    
    # Average loss for the epoch
    train_loss /= batch_count
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_batch_count = 0
    
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            loss = model(data.x, data.edge_index, data.batch)
            
            # Only count valid losses
            if not torch.isnan(loss) and not torch.isinf(loss):
                val_loss += loss.item()
                val_batch_count += 1
    
    # Average validation loss if we have valid batches
    if val_batch_count > 0:
        val_loss /= val_batch_count
        validation_losses.append(val_loss)
        
        # Update scheduler based on validation loss
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'graph_vae_model_improved.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    # Print progress
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}, Beta: {model.beta:.4f}")

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')  # Use log scale for better visualization
plt.legend()
plt.savefig('figures/improved_loss_plot.png')
plt.close()

print("Training completed. Loading best model...")

# Load the best model
model.load_state_dict(torch.load('graph_vae_model_improved.pt'))
model.eval()

# Sample graphs
print("Sampling graphs from trained model...")
with torch.no_grad():
    n_samples = 10
    sampled_adjs = model.sample(n_samples).cpu()

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
    
    # Check if the graph is disconnected using DFS
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

print(f"  Disconnected Graphs: {disconnected}/{n_samples} ({disconnected/n_samples*100:.1f}%)")

# Create a figure to visualize some of the sampled graphs
plt.figure(figsize=(15, 6))

for i in range(min(5, n_samples)):
    plt.subplot(1, 5, i+1)
    adj = sampled_adjs[i].numpy()
    # Get number of active nodes
    n_nodes = int(non_zeros[i].item())
    # Extract subgraph with actual nodes
    subgraph = adj[:n_nodes, :n_nodes]
    plt.imshow(subgraph, cmap='Blues')
    plt.title(f"Graph {i+1}\n{n_nodes} nodes")
    plt.axis('off')

plt.tight_layout()
plt.savefig('figures/improved_samples.png')
plt.close()

print("\nTraining and sampling completed!")
print("Results saved to 'figures/improved_loss_plot.png' and 'figures/improved_samples.png'")
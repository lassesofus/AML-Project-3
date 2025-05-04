import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# Helper function for interactive plots
def drawnow():
    """Force draw the current plot."""
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()

def plot_curves(train_history, val_history, epoch, save_dir='figures'):
    """
    Plot training and validation curves for different metrics
    
    Args:
        train_history: Dictionary of training metrics
        val_history: Dictionary of validation metrics
        epoch: Current epoch number
        save_dir: Directory to save plot images
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure for loss curves
    plt.figure('Loss', figsize=(12, 5))
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(train_history['total'], label='Train Total Loss')
    plt.plot(val_history['total'], label='Val Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.title('Total Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_history['reconstruction'], label='Recon')
    plt.plot(train_history['kl'], label='KL')
    plt.plot(train_history['structure_penalty'], label='Structure Penalty')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Component')
    plt.legend()
    plt.yscale('log')
    plt.title('Loss Components')
    plt.tight_layout()
    drawnow()
    
    # Create figure for penalties
    plt.figure('Penalties', figsize=(12, 5))
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(train_history['degree_dist'], label='Degree Dist')
    plt.plot(train_history['connectivity'], label='Connectivity')
    plt.xlabel('Epoch')
    plt.ylabel('Penalty Value')
    plt.legend()
    plt.title('Distribution & Connectivity Penalties')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_history['valency'], label='Valency')
    plt.plot(train_history['sparsity'], label='Sparsity')
    plt.xlabel('Epoch')
    plt.ylabel('Penalty Value')
    plt.legend()
    plt.title('Valency & Sparsity Penalties')
    plt.tight_layout()
    drawnow()
    
    # Create figure for disconnected graph ratio
    plt.figure('Stats', figsize=(12, 5))
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(val_history['disconnected_ratio'], label='Disconnected')
    plt.ylabel('Ratio')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.title('Disconnected Graph Ratio')
    
    # Plot latest degree distribution comparison
    plt.subplot(1, 2, 2)
    width = 0.35
    x = np.arange(7)
    
    target_dist = val_history['target_degree_dist'][-1].cpu().numpy()
    pred_dist = val_history['pred_degree_dist'][-1].cpu().numpy()
    
    plt.bar(x - width/2, target_dist, width, label='Target')
    plt.bar(x + width/2, pred_dist, width, label='Predicted')
    plt.xticks(x, ['0', '1', '2', '3', '4', '5', '6+'])
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Degree Distribution')
    plt.tight_layout()
    drawnow()
    
    # Create detailed comparison of train vs validation losses
    plt.figure('Train vs Val Losses', figsize=(14, 10))
    plt.clf()
    
    # Total loss comparison
    plt.subplot(2, 3, 1)
    plt.plot(train_history['total'], label='Train')
    plt.plot(val_history['total'], label='Val')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.yscale('log')
    plt.legend()
    
    # Reconstruction loss
    plt.subplot(2, 3, 2)
    plt.plot(train_history['reconstruction'], label='Train')
    plt.plot(val_history['reconstruction'], label='Val')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.yscale('log')
    plt.legend()
    
    # KL loss
    plt.subplot(2, 3, 3)
    plt.plot(train_history['kl'], label='Train')
    plt.plot(val_history['kl'], label='Val')
    plt.title('KL Divergence')
    plt.xlabel('Epoch')
    plt.yscale('log')
    plt.legend()
    
    # Degree distribution loss
    plt.subplot(2, 3, 4)
    plt.plot(train_history['degree_dist'], label='Train')
    plt.plot(val_history['degree_dist'], label='Val')
    plt.title('Degree Distribution Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Valency loss
    plt.subplot(2, 3, 5)
    plt.plot(train_history['valency'], label='Train')
    plt.plot(val_history['valency'], label='Val')
    plt.title('Valency Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Connectivity loss
    plt.subplot(2, 3, 6)
    plt.plot(train_history['connectivity'], label='Train')
    plt.plot(val_history['connectivity'], label='Val')
    plt.title('Connectivity Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    drawnow()
    
    # Save the figures periodically
    if epoch % 50 == 0 or epoch == 0:
        plt.figure('Loss')
        plt.savefig(f'{save_dir}/vgae_loss_{epoch}.png')
        plt.figure('Penalties')
        plt.savefig(f'{save_dir}/vgae_penalties_{epoch}.png')
        plt.figure('Stats')
        plt.savefig(f'{save_dir}/vgae_stats_{epoch}.png')
        plt.figure('Train vs Val Losses')
        plt.savefig(f'{save_dir}/vgae_detailed_losses_{epoch}.png')

def plot_histograms(original_metrics, vae_metrics, filename='method_comparison.png'):
    """
    Plot histograms comparing original and VAE-generated graph metrics
    
    Args:
        original_metrics: Metrics from original dataset (degrees, clustering, eigenvector, num_nodes, disconnected_count)
        vae_metrics: Metrics from VAE-generated graphs
        filename: Output filename for the plot
    """
    method_names = ['Original', 'Graph VAE']
    metric_names = ['Degree', 'Clustering', 'Eigenvector']
    all_metrics = [original_metrics[:3], vae_metrics[:3]]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            data = all_metrics[i][j]
            if not data:
                ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', 
                      verticalalignment='center', transform=ax.transAxes)
                continue

            if j == 0:
                max_deg = int(np.max(data)) if data else 0
                bins = np.arange(0, max_deg + 2) - 0.5
            else:
                bins = np.linspace(min(data) if data else 0, max(data) if data else 1, 21)

            ax.hist(data, bins=bins, density=True, alpha=0.7, label=method_names[i])
            ax.set_title(metric_names[j])
            ax.set_xlabel('Value')
            ax.set_ylabel('Density' if j == 0 else '')
            ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved histogram comparison to {filename}")

def plot_sample_graphs(original_graphs, vae_graphs, filename='sample_graphs_comparison.png'):
    """
    Plot sample graphs from original dataset and VAE-generated graphs
    
    Args:
        original_graphs: List of original NetworkX graphs
        vae_graphs: List of VAE-generated NetworkX graphs
        filename: Output filename for the plot
    """
    import networkx as nx
    import random
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, graphs in enumerate([original_graphs, vae_graphs]):
        num_to_sample = min(3, len(graphs))
        if num_to_sample == 0:
            for j in range(3):
                ax = axes[i, j]
                ax.text(0.5, 0.5, 'No Graphs', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{'Original' if i == 0 else 'VAE'} Sample {j+1}")
            continue

        samples = random.sample(graphs, num_to_sample)
        for j in range(3):
            ax = axes[i, j]
            if j < len(samples):
                G = samples[j]
                pos = nx.spring_layout(G, seed=42) if G.number_of_nodes() > 0 else {}
                nx.draw(G, pos, node_size=50, ax=ax, width=0.5)
                title = 'Original' if i == 0 else 'VAE'
                ax.set_title(f"{title} Sample {j+1} (N={G.number_of_nodes()})")
            else:
                ax.set_title(f"{'Original' if i == 0 else 'VAE'} Sample {j+1}")
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved sample graph comparison to {filename}")
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
    Plot sample graphs from original dataset and VAE-generated graphs.
    Shows the full graphs without filtering, including isolated nodes.
    
    Args:
        original_graphs: List of original NetworkX graphs
        vae_graphs: List of VAE-generated NetworkX graphs
        filename: Output filename for the plot
    """
    import networkx as nx
    import random
    
    # Increase number of samples to show more variety
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    plt.subplots_adjust(hspace=0.5)  # Add more space between rows
    
    # Select one graph with multiple components if available (for VAE)
    multi_component_graphs = [g for g in vae_graphs if nx.number_connected_components(g) > 1]
    
    for i, graphs in enumerate([original_graphs, vae_graphs]):
        title_prefix = 'Original' if i == 0 else 'VAE'
        
        # For VAE row, try to choose diverse samples
        if i == 1 and multi_component_graphs:
            # Select one multi-component graph if available, and random others
            samples = [multi_component_graphs[0]] if multi_component_graphs else []
            remaining = [g for g in vae_graphs if g not in samples]
            num_random = min(3, len(remaining))
            if num_random > 0:
                samples.extend(random.sample(remaining, num_random))
        else:
            # For original graphs or if no multi-component, just random sample
            num_to_sample = min(4, len(graphs))
            if num_to_sample == 0:
                samples = []
            else:
                samples = random.sample(graphs, num_to_sample)
        
        # Draw each selected graph
        for j in range(4):
            ax = axes[i, j]
            if j < len(samples):
                G = samples[j]
                
                # Get stats for the title
                n_nodes = G.number_of_nodes()
                n_edges = G.number_of_edges()
                n_components = nx.number_connected_components(G)
                n_isolates = len(list(nx.isolates(G)))
                
                # Create layout that clearly shows all nodes including isolates
                if n_nodes > 0:
                    if n_isolates > 0:
                        # Special layout to better visualize isolated nodes
                        pos = nx.spring_layout(G, seed=42+j)
                        # Adjust isolated nodes to be more visible
                        isolates = list(nx.isolates(G))
                        for idx, node in enumerate(isolates):
                            angle = 2 * np.pi * idx / len(isolates)
                            pos[node] = np.array([1.5 * np.cos(angle), 1.5 * np.sin(angle)])
                    else:
                        pos = nx.spring_layout(G, seed=42+j)
                else:
                    pos = {}
                
                # Draw the graph with node colors indicating connected components
                components = list(nx.connected_components(G))
                colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(components))))
                
                # Assign colors to nodes based on their component
                node_colors = []
                for node in G.nodes():
                    # Find which component this node belongs to
                    for idx, component in enumerate(components):
                        if node in component:
                            node_colors.append(colors[idx % len(colors)])
                            break
                    else:
                        # This should not happen, but just in case
                        node_colors.append('lightgray')
                
                # Draw graph with additional details
                nx.draw(G, pos, node_size=80, ax=ax, width=0.8, 
                        node_color=node_colors, 
                        with_labels=n_nodes < 20)  # Only show labels for small graphs
                
                # Detailed title with graph statistics
                ax.set_title(f"{title_prefix} #{j+1}: {n_nodes} nodes, {n_edges} edges\n"
                             f"{n_components} comp{'s' if n_components != 1 else ''}, "
                             f"{n_isolates} isolated")
            else:
                ax.set_title(f"{title_prefix} Sample {j+1}")
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved sample graph comparison to {filename}")
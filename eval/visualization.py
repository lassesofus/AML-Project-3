import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import numpy as np
import torch
import seaborn as sns
from torch_geometric.utils import to_networkx

def plot_sampled_graphs_collage(sampled_graphs, rows=3, cols=3, figsize=(15, 15), save_path='sampled_graphs_collage.pdf'):
    """
    Create a collage of sampled graphs.
    
    Parameters:
    -----------
    sampled_graphs : list
        List of sampled graphs to visualize
    rows : int
        Number of rows in the collage
    cols : int
        Number of columns in the collage
    figsize : tuple
        Figure size (width, height)
    save_path : str
        Path to save the figure
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    for i, ax in enumerate(axes.flatten()):
        if i < len(sampled_graphs):
            graph_nx = nx.from_numpy_array(sampled_graphs[i].cpu().numpy())
            nx.draw(graph_nx, ax=ax, node_size=40, with_labels=False, arrows=False)
            ax.set_title(f"Graph {i+1}", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved collage to {save_path}")

def plot_graph_statistics(training_degrees, training_clustering, training_eigenvector,
                          erdos_renyi_degrees, erdos_renyi_clustering, erdos_renyi_eigenvector,
                          generated_degrees, generated_clustering, generated_eigenvector,
                          save_path='graph_statistics_comparison_columns.png'):
    """Plot histograms of graph statistics for comparison."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Set titles for columns
    axes[0, 0].set_title('Node Degree')
    axes[0, 1].set_title('Clustering Coefficient')
    axes[0, 2].set_title('Eigenvector Centrality')
    
    # Row labels
    row_labels = ['Training', 'Erdős-Rényi', 'VAE']
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label)
    
    # Colors for different data sources
    colors = ['blue', 'green', 'red']
    
    # Group data for iteration
    all_degrees = [training_degrees, erdos_renyi_degrees, generated_degrees]
    all_clustering = [training_clustering, erdos_renyi_clustering, generated_clustering]
    all_eigenvector = [training_eigenvector, erdos_renyi_eigenvector, generated_eigenvector]
    
    for i, (degrees, clustering, eigenvector) in enumerate(zip(all_degrees, all_clustering, all_eigenvector)):
        color = colors[i]
        
        # Plot node degree
        axes[i, 0].hist(degrees, bins=20, color=color, alpha=0.7, density=True)
        
        # Handle clustering coefficient - Split into zero and non-zero values
        total_count = len(clustering)

        # Add non-zero values as a histogram on same plot if they exist

        # Count non-zeros in bins
        hist, bin_edges = np.histogram(clustering, bins=5, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Normalize to get density
        hist = hist / total_count
        
        # Plot as bars with consistent scaling
        bin_width = bin_edges[1] - bin_edges[0]
        axes[i, 1].bar(bin_centers, hist, width=bin_width*0.8, 
                        color=color, alpha=0.7, label='Non-zero')
    
        # Add legend, set limits and labels
        axes[i, 1].legend(loc='upper right')
        axes[i, 1].set_xlim(-0.05, 1.05)
        # axes[i, 1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        # Plot eigenvector centrality
        axes[i, 2].hist(eigenvector, bins=20, color=color, alpha=0.7, density=True)
        
        # Set labels for bottom row
        if i == 2:
            axes[i, 0].set_xlabel('Degree')
            axes[i, 1].set_xlabel('Clustering Coefficient')
            axes[i, 2].set_xlabel('Eigenvector Centrality')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved graph statistics to {save_path}")


def plot_graph_comparison(training_samples, erdos_samples, vae_samples, save_path='graph_comparison.png', 
                         grid_height_ratios=[10, 10, 10, 1],
                         grid_hspace=0.3,
                         sep_line1_y=0.69,
                         sep_line2_y=0.36,
                         train_label_y=0.83,
                         erdos_label_y=0.5,
                         vae_label_y=0.17,
                         tight_layout_rect=[0.03, 0, 1, 0.98]):
    """
    Visualize and compare graphs from training data, Erdős-Rényi model, and VAE.
    
    Parameters:
    -----------
    training_samples : list
        List of 3 graphs from the training dataset
    erdos_samples : list
        List of 3 graphs from the Erdős-Rényi model
    vae_samples : list
        List of 3 graphs from the VAE model
    save_path : str
        Path to save the figure
    grid_height_ratios : list
        Height ratios for the rows in the GridSpec
    grid_hspace : float
        Horizontal spacing between subplots
    sep_line1_y, sep_line2_y : float
        Y-positions of separator lines
    train_label_y, erdos_label_y, vae_label_y : float
        Y-positions of row labels
    tight_layout_rect : list
        Rectangle for tight_layout adjustment
    """
    # Create figure with grid - adjusted height ratios for better spacing
    fig = plt.figure(figsize=(15, 15))
    gs = plt.GridSpec(4, 3, height_ratios=grid_height_ratios, hspace=grid_hspace)
    
    # Prepare all graphs for consistent coloring
    all_graphs = []
    for graph in training_samples:
        if not isinstance(graph, nx.Graph):
            graph = to_networkx(graph, to_undirected=True)
        all_graphs.append(graph)
        
    for graph in erdos_samples:
        if isinstance(graph, torch.Tensor):
            graph = nx.from_numpy_array(graph.cpu().numpy())
        all_graphs.append(graph)
        
    for graph in vae_samples:
        if isinstance(graph, torch.Tensor):
            graph = nx.from_numpy_array(graph.cpu().numpy())
        all_graphs.append(graph)
    
    # Find global max degree across all graphs
    global_max_degree = 0
    for G in all_graphs:
        degrees = [d for _, d in G.degree()]
        global_max_degree = max(global_max_degree, max(degrees) if degrees else 0)
    
    # Create a discrete colormap with integer steps
    bounds = np.arange(0, global_max_degree + 2) - 0.5  # Boundaries between colors
    cmap = plt.cm.viridis
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    # Add row titles as bold text
    fig.text(0.01, train_label_y, "Training Graphs", va="center", ha="left", 
             fontsize=12, fontweight="bold", rotation="vertical")
    fig.text(0.01, erdos_label_y, "Erdős-Rényi Graphs", va="center", ha="left", 
             fontsize=12, fontweight="bold", rotation="vertical")
    fig.text(0.01, vae_label_y, "VAE Graphs", va="center", ha="left", 
             fontsize=12, fontweight="bold", rotation="vertical")
    
    # Plot training graphs (top row)
    for i, graph in enumerate(training_samples):
        ax = fig.add_subplot(gs[0, i])
        
        if not isinstance(graph, nx.Graph):
            graph = to_networkx(graph, to_undirected=True)
            
        node_degrees = dict(graph.degree())
        node_colors = [node_degrees[n] for n in graph.nodes()]
        
        # Use spring layout with fixed parameters for more consistent sizing
        pos = nx.spring_layout(graph, seed=42)
        nx.draw(graph, pos=pos, ax=ax, node_size=40, with_labels=False,
                node_color=node_colors, arrows=False, cmap=cmap,
                vmin=0, vmax=global_max_degree)
        
        ax.set_title(f"Graph {i+1}", fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # Add horizontal separator line between training and Erdős-Rényi rows
    fig.add_artist(plt.Line2D([0.05, 0.95], [sep_line1_y, sep_line1_y], color='black', 
                             linewidth=1, transform=fig.transFigure))
    
    # Plot Erdos-Renyi graphs (middle row)
    for i, graph in enumerate(erdos_samples):
        ax = fig.add_subplot(gs[1, i])
        
        if isinstance(graph, torch.Tensor):
            graph = nx.from_numpy_array(graph.cpu().numpy())
            
        node_degrees = dict(graph.degree())
        node_colors = [node_degrees[n] for n in graph.nodes()]
        
        pos = nx.spring_layout(graph, seed=42)
        nx.draw(graph, pos=pos, ax=ax, node_size=40, with_labels=False,
                node_color=node_colors, arrows=False, cmap=cmap,
                vmin=0, vmax=global_max_degree)
        
        ax.set_title(f"Graph {i+1}", fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # Add horizontal separator line between Erdős-Rényi and VAE rows
    fig.add_artist(plt.Line2D([0.05, 0.95], [sep_line2_y, sep_line2_y], color='black', 
                             linewidth=1, transform=fig.transFigure))
    
    # Plot VAE graphs (bottom row)
    for i, graph in enumerate(vae_samples):
        ax = fig.add_subplot(gs[2, i])
        
        if isinstance(graph, torch.Tensor):
            graph = nx.from_numpy_array(graph.cpu().numpy())
            
        node_degrees = dict(graph.degree())
        node_colors = [node_degrees[n] for n in graph.nodes()]
        
        pos = nx.spring_layout(graph, seed=42)
        nx.draw(graph, pos=pos, ax=ax, node_size=40, with_labels=False,
                node_color=node_colors, arrows=False, cmap=cmap,
                vmin=0, vmax=global_max_degree)
        
        ax.set_title(f"Graph {i+1}", fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # Add discrete colorbar with integer ticks
    cbar_ax = fig.add_subplot(gs[3, :])
    cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                                  orientation='horizontal', label='Node Degree')
    
    # Set integer ticks at the center of each color segment
    cb.set_ticks(range(0, global_max_degree + 1))
    cb.set_ticklabels(range(0, global_max_degree + 1))
    
    # Make colorbar label bold
    cb.set_label('Node Degree', fontweight='bold', fontsize=11)
    
    # Adjust the layout with more padding
    plt.tight_layout(rect=tight_layout_rect)
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved graph comparison to {save_path}")

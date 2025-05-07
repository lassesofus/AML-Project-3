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

def plot_degree_distributions(axes, all_degrees, colors, row_labels):
    """
    Plot node degree distributions for different graph sources.
    
    Parameters:
    -----------
    axes : list of matplotlib.axes
        List of axes to plot on
    all_degrees : list of lists
        Lists of degree values for each graph source
    colors : list
        Colors to use for each graph source
    row_labels : list
        Labels for each graph source
    """
    # Calculate histograms first to find the maximum y-value
    max_density = 0
    histograms = []
    
    for degrees in all_degrees:
        bins = np.arange(0, 11) - 0.5  # Edges at -0.5, 0.5, 1.5, ..., 9.5
        hist, _ = np.histogram(degrees, bins=bins, density=True)
        histograms.append(hist)
        max_density = max(max_density, np.max(hist)) if len(hist) > 0 else max_density
    
    # Add a small margin to the maximum density
    max_density *= 1.1
    
    for i, degrees in enumerate(all_degrees):
        color = colors[i]
        
        # Create fixed bins from 0 to 9 with integer steps (no gaps)
        bins = np.arange(0, 11) - 0.5  # Edges at -0.5, 0.5, 1.5, ..., 9.5
        
        # Plot node degree histogram with fixed bins
        axes[i].hist(degrees, bins=bins, color=color, alpha=0.7, density=True, 
                    rwidth=1.0,  edgecolor='black', linewidth=0.5)  # Added black edges
          # rwidth=1.0 ensures no gaps between bars
        
        # Set x-axis limits and ticks consistently for all plots
        axes[i].set_xlim(-0.5, 9.5)
        axes[i].set_xticks(range(0, 10))
        axes[i].tick_params(axis='x', labelsize=14)
        
        # Set consistent y-axis limits
        axes[i].set_ylim(0, max_density)

def plot_clustering_distributions(axes, all_clustering, colors, row_labels):
    """
    Plot clustering coefficient distributions for different graph sources.
    
    Parameters:
    -----------
    axes : list of matplotlib.axes
        List of axes to plot on
    all_clustering : list of lists
        Lists of clustering coefficient values for each graph source
    colors : list
        Colors to use for each graph source
    row_labels : list
        Labels for each graph source
    """
    # Calculate histograms first to find the maximum y-value
    max_density = 0
    num_bins = 10
    bin_edges = np.linspace(0, 1, num_bins + 1)
    
    for clustering in all_clustering:
        total_count = len(clustering)
        hist, _ = np.histogram(clustering, bins=bin_edges)
        hist = hist / total_count if total_count > 0 else hist
        max_density = max(max_density, np.max(hist)) if len(hist) > 0 else max_density
    
    # Add a small margin to the maximum density
    max_density *= 1.1
    
    for i, clustering in enumerate(all_clustering):
        color = colors[i]
        
        # Use finer bins for clustering coefficients (0 to 1)
        num_bins = 10  # Increased from 5 to 10 for finer granularity
        bin_edges = np.linspace(0, 1, num_bins + 1)  # Equally spaced bins from 0 to 1
        
        # Count values in bins and normalize
        total_count = len(clustering)
        hist, _ = np.histogram(clustering, bins=bin_edges)
        hist = hist / total_count if total_count > 0 else hist
        
        # Plot bars aligned with bin edges instead of centered, with black edges
        bin_width = bin_edges[1] - bin_edges[0]
        axes[i].bar(bin_edges[:-1], hist, width=bin_width, 
                   color=color, alpha=0.7,
                   align='edge', edgecolor='black', linewidth=0.5)  # Added black edges
        
        # Set consistent x limits and tick marks at more readable intervals
        axes[i].set_xlim(0, 1.0)
        # Show fewer tick marks to avoid crowding
        tick_positions = np.linspace(0, 1, 6)  # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        axes[i].set_xticks(tick_positions)
        axes[i].set_xticklabels([f'{x:.1f}' for x in tick_positions])
        axes[i].tick_params(axis='x', labelsize=14)
        
        # Set consistent y-axis limits
        axes[i].set_ylim(0, max_density)

def plot_eigenvector_distributions(axes, all_eigenvector, colors, row_labels):
    """
    Plot eigenvector centrality distributions for different graph sources.
    
    Parameters:
    -----------
    axes : list of matplotlib.axes
        List of axes to plot on
    all_eigenvector : list of lists
        Lists of eigenvector centrality values for each graph source
    colors : list
        Colors to use for each graph source
    row_labels : list
        Labels for each graph source
    """
    # Calculate histograms first to find the maximum y-value
    max_density = 0
    
    for eigenvector in all_eigenvector:
        if eigenvector:
            hist, _ = np.histogram(eigenvector, bins=20, density=True)
            max_density = max(max_density, np.max(hist)) if len(hist) > 0 else max_density
    
    # Add a small margin to the maximum density
    max_density *= 1.1
    
    for i, eigenvector in enumerate(all_eigenvector):
        color = colors[i]
        
        # Plot eigenvector centrality histogram
        axes[i].hist(eigenvector, bins=20, color=color, alpha=0.7, density=True, 
                    edgecolor='black', linewidth=0.5)  # Added black edges
        
        # Set appropriate ticks for eigenvector centrality histogram
        if eigenvector:
            max_eigen = max(eigenvector)
            min_eigen = min(eigenvector)
            eigen_ticks = np.linspace(min_eigen, max_eigen, 6)
            axes[i].set_xticks(eigen_ticks)
            axes[i].set_xticklabels([f'{x:.2f}' for x in eigen_ticks])
            axes[i].tick_params(axis='x', labelsize=14)
        
        # Set consistent y-axis limits
        axes[i].set_ylim(0, max_density)

def plot_graph_statistics(training_degrees, training_clustering, training_eigenvector,
                          erdos_renyi_degrees, erdos_renyi_clustering, erdos_renyi_eigenvector,
                          generated_degrees, generated_clustering, generated_eigenvector,
                          save_path='graph_statistics_comparison_columns.png'):
    """Plot histograms of graph statistics for comparison."""
    # Create figure with more left padding for labels
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    # Set titles for columns
    axes[0, 0].set_title('Node Degree', fontweight='bold', fontsize=24)
    axes[0, 1].set_title('Clustering Coefficient', fontweight='bold', fontsize=24)
    axes[0, 2].set_title('Eigenvector Centrality', fontweight='bold', fontsize=24)
    
    # Row labels
    row_labels = ['Training', 'Erdős-Rényi', 'VAE']
    
    # Add row labels as bold text on the left side
    for i, label in enumerate(row_labels):
        # Position the labels within the figure boundary
        fig.text(0.013, 0.8 - i*0.3, label, fontweight='bold', ha='center', va='center', 
                rotation='vertical', fontsize=24)
    
        # Add global "Density" label
        fig.text(0.033, 0.8 - i*0.3, 'Density', ha='center', va='center', 
                rotation='vertical', fontsize=16)
    
    # Colors for different data sources
    colors = ['blue', 'green', 'red']
    
    # Group data for each type of statistic
    all_degrees = [training_degrees, erdos_renyi_degrees, generated_degrees]
    all_clustering = [training_clustering, erdos_renyi_clustering, generated_clustering]
    all_eigenvector = [training_eigenvector, erdos_renyi_eigenvector, generated_eigenvector]
    
    # Plot each type of statistic using specialized functions
    plot_degree_distributions(axes[:, 0], all_degrees, colors, row_labels)
    plot_clustering_distributions(axes[:, 1], all_clustering, colors, row_labels)
    plot_eigenvector_distributions(axes[:, 2], all_eigenvector, colors, row_labels)
    
    # Set bottom row labels
    # axes[2, 0].set_xlabel('Degree')
    # axes[2, 1].set_xlabel('Clustering Coefficient')
    # axes[2, 2].set_xlabel('Eigenvector Centrality')
    
    plt.tight_layout(rect=[0.038, 0, 1, 1])  # Adjust layout but preserve left space for labels
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

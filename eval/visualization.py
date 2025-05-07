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
    """
    Plot histogram comparisons of graph statistics.
    
    Parameters:
    -----------
    training_* : list
        Statistics from training graphs
    erdos_renyi_* : list
        Statistics from Erdős-Rényi graphs
    generated_* : list
        Statistics from VAE-generated graphs
    save_path : str
        Path to save the figure
    """
    # Filter eigenvector centrality values to only include those > 0
    training_eigenvector = [val for val in training_eigenvector if val > 0]
    erdos_renyi_eigenvector = [val for val in erdos_renyi_eigenvector if val > 0]
    generated_eigenvector = [val for val in generated_eigenvector if val > 0]

    # Define bins
    num_bins = 15
    degree_min = int(min(training_degrees + generated_degrees + erdos_renyi_degrees))
    degree_max = 8  # Fixed upper limit for node degree at 8
    degree_bins = list(range(degree_min, degree_max + 1))  # Integer bins from min to max

    # Create custom bins for clustering coefficient with explicit first bin for zero values
    cluster_max = max(training_clustering + generated_clustering + erdos_renyi_clustering)
    # Special bin structure with a narrow bin for zeros then regular bins
    clustering_bins = [0, 0.001] + list(np.linspace(0.001, cluster_max, 7))

    # Calculate bin centers for x-ticks
    clustering_ticks = [0] + list(np.linspace(0.1, cluster_max, 6))
    clustering_labels = ['0'] + [f'{x:.2f}' for x in np.linspace(0.1, cluster_max, 6)]

    # Create a 3-by-3 grid of subplots with shared y-axis per column
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharey='col')

    # Compute global axis limits for each statistic
    clustering_min, clustering_max = min(training_clustering + generated_clustering + erdos_renyi_clustering), max(training_clustering + generated_clustering + erdos_renyi_clustering)
    eigenvector_min, eigenvector_max = min(training_eigenvector + generated_eigenvector + erdos_renyi_eigenvector), max(training_eigenvector + generated_eigenvector + erdos_renyi_eigenvector)

    # Add column headers
    axes[0, 0].set_title('Node Degree', fontsize=14, fontweight='bold')
    axes[0, 1].set_title('Clustering Coefficient', fontsize=14, fontweight='bold')
    axes[0, 2].set_title('Eigenvector Centrality', fontsize=14, fontweight='bold')

    # Add row labels
    fig.text(0.01, 0.83, 'Training', fontsize=14, fontweight='bold', rotation=90, va='center')
    fig.text(0.01, 0.5, 'Erdős-Rényi', fontsize=14, fontweight='bold', rotation=90, va='center')
    fig.text(0.01, 0.17, 'VAE', fontsize=14, fontweight='bold', rotation=90, va='center')

    # Add individual density labels for each row
    fig.text(0.06, 0.83, 'Density', fontsize=12, rotation=90, va='center')
    fig.text(0.06, 0.5, 'Density', fontsize=12, rotation=90, va='center')
    fig.text(0.06, 0.17, 'Density', fontsize=12, rotation=90, va='center')

    # Add individual value labels for each column
    fig.text(0.2, 0.02, 'Value', fontsize=12, ha='center')
    fig.text(0.5, 0.02, 'Value', fontsize=12, ha='center')
    fig.text(0.8, 0.02, 'Value', fontsize=12, ha='center')

    # Plot all histograms with fewer labels and larger bars
    # Row 1: Training Graphs
    sns.histplot(training_degrees, color='blue', kde=False, stat="density", bins=degree_bins, ax=axes[0, 0], 
                line_kws={"linewidth": 2}, element="bars", alpha=0.7, discrete=True)
    sns.histplot(training_clustering, color='blue', kde=False, stat="density", bins=clustering_bins, ax=axes[0, 1], 
                line_kws={"linewidth": 2}, element="bars", alpha=0.7)
    axes[0, 1].set_xticks(clustering_ticks)
    axes[0, 1].set_xticklabels(clustering_labels, rotation=45, ha='center')
    sns.histplot(training_eigenvector, color='blue', kde=False, stat="density", bins=num_bins, ax=axes[0, 2], 
                line_kws={"linewidth": 2}, element="bars", alpha=0.7)

    # Row 2: Erdős-Rényi
    sns.histplot(erdos_renyi_degrees, color='red', kde=False, stat="density", bins=degree_bins, ax=axes[1, 0], 
                line_kws={"linewidth": 2}, element="bars", alpha=0.7, discrete=True)
    sns.histplot(erdos_renyi_clustering, color='red', kde=False, stat="density", bins=clustering_bins, ax=axes[1, 1], 
                line_kws={"linewidth": 2}, element="bars", alpha=0.7)
    axes[1, 1].set_xticks(clustering_ticks)
    axes[1, 1].set_xticklabels(clustering_labels, rotation=45, ha='center')
    sns.histplot(erdos_renyi_eigenvector, color='red', kde=False, stat="density", bins=num_bins, ax=axes[1, 2], 
                line_kws={"linewidth": 2}, element="bars", alpha=0.7)

    # Row 3: VAE Generated Graphs
    sns.histplot(generated_degrees, color='green', kde=False, stat="density", bins=degree_bins, ax=axes[2, 0], 
                line_kws={"linewidth": 2}, element="bars", alpha=0.7, discrete=True)
    sns.histplot(generated_clustering, color='green', kde=False, stat="density", bins=clustering_bins, ax=axes[2, 1], 
                line_kws={"linewidth": 2}, element="bars", alpha=0.7)
    axes[2, 1].set_xticks(clustering_ticks)
    axes[2, 1].set_xticklabels(clustering_labels, rotation=45, ha='center')
    sns.histplot(generated_eigenvector, color='green', kde=False, stat="density", bins=num_bins, ax=axes[2, 2], 
                line_kws={"linewidth": 2}, element="bars", alpha=0.7)

    # Set consistent x-axis limits for each column
    for ax in [axes[0, 0], axes[1, 0], axes[2, 0]]:
        ax.set_xlim(degree_min - 0.5, degree_max + 0.5)  # Extend slightly beyond the integer values
        ax.set_xticks(degree_bins)  # Set x-ticks to be exactly the integer bins
        ax.tick_params(labelsize=12)
        ax.set_ylabel('')  # Remove individual y labels
        
    for ax in [axes[0, 1], axes[1, 1], axes[2, 1]]:
        ax.set_xlim(-0.05, cluster_max + 0.05)  # Extend left boundary to show full first bar
        ax.tick_params(labelsize=12)
        ax.set_ylabel('')  # Remove individual y labels
        # Fix y-axis to reasonable density values
        ax.set_ylim(0, None)  # Let matplotlib determine upper bound based on data
        
    for ax in [axes[0, 2], axes[1, 2], axes[2, 2]]:
        ax.set_xlim(max(0.01, eigenvector_min), eigenvector_max)  # Start from slightly above 0
        ax.tick_params(labelsize=12) 
        ax.set_ylabel('')  # Remove individual y labels
        ax.set_ylim(0, None)  # Let matplotlib determine upper bound based on data

    # Ensure density scaling is correct across all plots
    for row in axes:
        for ax in row:
            current_ylim = ax.get_ylim()
            if current_ylim[1] > 5:  # If scale appears to be percentage rather than density
                ax.set_ylim(0, min(5, current_ylim[1]/100))  # Cap at 5 for density scale

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0.08, 0.04, 1, 0.97])  # Adjust to leave space for labels
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
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

# %% 
import pdb
import torch
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree
from erdos_renyi import ErdosRenyiSampler
from architecture import get_vae
from utils import load_model_checkpoint
import matplotlib as mpl

# Set a fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f"Using seed: {SEED}")

device = 'cpu'

# Configs
device = 'cpu'
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
node_feature_dim = dataset.num_node_features  # should be 7
latent_dim = 16
batch_size = 16
model_path = './models/graph_vae.pt'


# Compute empirical distribution of the number of nodes in the dataset
num_nodes_list = [data.num_nodes for data in dataset]

vae = get_vae(num_nodes_list=num_nodes_list)

vae = load_model_checkpoint(vae, model_path)

# Sample N graphs using the generator
N = 1000  # Number of graphs to sample

erdos = ErdosRenyiSampler(dataset)  # Adjust p as needed

def compute_graph_metrics(candidate_graphs, reference_graphs):
    """
    Compute novelty, uniqueness, and novel & unique metrics for candidate graphs.
    
    Parameters:
    -----------
    candidate_graphs : list
        List of graphs to evaluate
    reference_graphs : list
        List of reference graphs to compare against
        
    Returns:
    --------
    tuple: (novel_percentage, unique_percentage, novel_and_unique_percentage)
    """
    # Convert to NetworkX graphs if needed
    candidate_graphs_nx = []
    for graph in candidate_graphs:
        if isinstance(graph, nx.Graph):
            candidate_graphs_nx.append(graph)
        elif isinstance(graph, torch.Tensor):
            candidate_graphs_nx.append(nx.from_numpy_array(graph.cpu().numpy()))
        else:
            candidate_graphs_nx.append(to_networkx(graph, to_undirected=True))
    
    reference_graphs_nx = []
    for graph in reference_graphs:
        if isinstance(graph, nx.Graph):
            reference_graphs_nx.append(graph)
        elif isinstance(graph, torch.Tensor):
            reference_graphs_nx.append(nx.from_numpy_array(graph.cpu().numpy()))
        else:
            reference_graphs_nx.append(to_networkx(graph, to_undirected=True))
    
    # Compute novelty for each graph
    novel_list = [
        not any(nx.is_isomorphic(g1, g2) for g2 in reference_graphs_nx)
        for g1 in candidate_graphs_nx
    ]
    novel_count = sum(novel_list)
    novel_percentage = novel_count / len(candidate_graphs_nx) * 100

    # Compute uniqueness - FIXED: Count distinct graph types instead of graphs that appear only once
    graph_hashes = [nx.weisfeiler_lehman_graph_hash(g) for g in candidate_graphs_nx]
    unique_hashes = set(graph_hashes)  # Get unique hashes
    unique_count = len(unique_hashes)  # Number of distinct graph types
    unique_percentage = unique_count / len(candidate_graphs_nx) * 100
    
    # Count occurrences of each hash (still needed for novel_and_unique calculation)
    hash_counts = {}
    for h in graph_hashes:
        hash_counts[h] = hash_counts.get(h, 0) + 1

    # Create a list of novel and unique graphs
    # A graph is novel_and_unique if it's novel and its type (hash) is unique
    novel_and_unique_hashes = set([h for h, is_novel in zip(graph_hashes, novel_list) if is_novel])
    novel_and_unique_count = len(novel_and_unique_hashes)
    novel_and_unique_percentage = novel_and_unique_count / len(candidate_graphs_nx) * 100

    return novel_percentage, unique_percentage, novel_and_unique_percentage

def evaluate_all_graph_sources(training_graphs, erdos_graphs, vae_graphs):
    """
    Evaluate novelty, uniqueness, and novel & unique metrics for all graph sources.
    
    Parameters:
    -----------
    training_graphs : list
        List of graphs from the training dataset
    erdos_graphs : list
        List of graphs from the Erdős-Rényi model
    vae_graphs : list
        List of graphs from the VAE model
        
    Returns:
    --------
    dict: Dictionary of metrics for each graph source
    """
    results = {}
    
    # Training vs. Itself (for uniqueness only)
    _, training_unique, _ = compute_graph_metrics(training_graphs, [])
    results["training"] = {"unique": training_unique}
    
    # Erdős-Rényi vs. Training
    erdos_novel, erdos_unique, erdos_novel_unique = compute_graph_metrics(erdos_graphs, training_graphs)
    results["erdos"] = {
        "novel": erdos_novel,
        "unique": erdos_unique,
        "novel_unique": erdos_novel_unique
    }
    
    # VAE vs. Training
    vae_novel, vae_unique, vae_novel_unique = compute_graph_metrics(vae_graphs, training_graphs)
    results["vae"] = {
        "novel": vae_novel,
        "unique": vae_unique,
        "novel_unique": vae_novel_unique
    }
    
    return results

def print_metrics_table(results):
    """
    Print metrics in a clean tabular format.
    
    Parameters:
    -----------
    results : dict
        Dictionary of metrics from evaluate_all_graph_sources
    """
    print("\n" + "="*60)
    print(f"{'Graph Source':<20} {'Novel (%)':<15} {'Unique (%)':<15} {'Novel & Unique (%)':<15}")
    print("-"*60)
    
    # Training data (uniqueness only)
    print(f"{'Training':<20} {'N/A':<15} {results['training']['unique']:<15.2f} {'N/A':<15}")
    
    # Erdős-Rényi
    print(f"{'Erdős-Rényi':<20} {results['erdos']['novel']:<15.2f} {results['erdos']['unique']:<15.2f} {results['erdos']['novel_unique']:<15.2f}")
    
    # VAE
    print(f"{'VAE':<20} {results['vae']['novel']:<15.2f} {results['vae']['unique']:<15.2f} {results['vae']['novel_unique']:<15.2f}")
    print("="*60)

z = torch.randn(N, latent_dim).to(device)

sampled_graphs = vae.sample(N)
print(f"Sampled {len(sampled_graphs)} graphs")
# Create a collage of 5x5 sampled graphs
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for i, ax in enumerate(axes.flatten()):
    if i < len(sampled_graphs):
        graph_nx = nx.from_numpy_array(sampled_graphs[i].cpu().numpy())
        nx.draw(graph_nx, ax=ax, node_size=40, with_labels=False, arrows=False)
        ax.set_title(f"Graph {i+1}", fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig('sampled_graphs_collage.pdf')
plt.close()

# Compute metrics
training_graphs = [data for data in dataset]
erdos_graphs = [g for g in erdos.sample_graphs(N) if g is not None]  # Filter out None values

# Evaluate all graph sources
print("\nCalculating graph metrics (this may take a while)...")
metrics_results = evaluate_all_graph_sources(training_graphs, erdos_graphs, sampled_graphs)
print_metrics_table(metrics_results)

import seaborn as sns
from torch_geometric.utils import to_networkx

def compute_graph_statistics(graphs):
    """Compute node degree, clustering coefficient, and eigenvector centrality for a list of graphs."""
    degree_list = []
    clustering_list = []
    eigenvector_centrality_list = []
    for graph in graphs:
        # Convert to NetworkX graph
        if isinstance(graph, nx.Graph):
            graph_nx = graph
        elif isinstance(graph, torch.Tensor):
            graph_nx = nx.from_numpy_array(graph.cpu().numpy())
        else:
            graph_nx = to_networkx(graph, to_undirected=True)

        # Compute node degree
        degrees = [d for _, d in graph_nx.degree()]
        degree_list.extend(degrees)

        # Compute clustering coefficient
        clustering = list(nx.clustering(graph_nx).values())
        clustering_list.extend(clustering)

        # Compute eigenvector centrality
        try:
            eigenvector_centrality = list(nx.eigenvector_centrality_numpy(graph_nx).values())
        except nx.NetworkXError:
            eigenvector_centrality = [0] * graph_nx.number_of_nodes()  # Handle disconnected graphs
        eigenvector_centrality_list.extend(eigenvector_centrality)

    return degree_list, clustering_list, eigenvector_centrality_list


# Compute statistics for the training graphs
training_degrees, training_clustering, training_eigenvector = compute_graph_statistics(training_graphs)

# Compute statistics for the generated graphs
generated_degrees, generated_clustering, generated_eigenvector = compute_graph_statistics(sampled_graphs)

# Compute statistics for the Erdős-Rényi baseline
erdos_renyi_degrees, erdos_renyi_clustering, erdos_renyi_eigenvector = compute_graph_statistics(erdos_graphs)

# Add this code to check the distribution of clustering coefficients
print("Clustering coefficient distribution analysis:")
print(f"Training - Zero values: {training_clustering.count(0)} out of {len(training_clustering)} ({100*training_clustering.count(0)/len(training_clustering):.2f}%)")
print(f"Erdős-Rényi - Zero values: {erdos_renyi_clustering.count(0)} out of {len(erdos_renyi_clustering)} ({100*erdos_renyi_clustering.count(0)/len(erdos_renyi_clustering):.2f}%)")
print(f"VAE - Zero values: {generated_clustering.count(0)} out of {len(generated_clustering)} ({100*generated_clustering.count(0)/len(generated_clustering):.2f}%)")

# Filter eigenvector centrality values to only include those > 0
training_eigenvector = [val for val in training_eigenvector if val > 0]
erdos_renyi_eigenvector = [val for val in erdos_renyi_eigenvector if val > 0]
generated_eigenvector = [val for val in generated_eigenvector if val > 0]

# Define fewer bins to make bars larger
num_bins = 15
# Define integer bins for node degree
degree_min = int(min(training_degrees + generated_degrees + erdos_renyi_degrees))
degree_max = 8  # Fixed upper limit for node degree at 8
degree_bins = list(range(degree_min, degree_max + 1))  # Integer bins from min to max

# Create custom bins for clustering coefficient with explicit first bin for zero values
cluster_max = max(training_clustering + generated_clustering + erdos_renyi_clustering)
# Special bin structure with a narrow bin for zeros then regular bins
clustering_bins = [0, 0.001] + list(np.linspace(0.001, cluster_max, 7))

# Calculate bin centers for x-ticks - ensure first bin is properly represented
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
# Special handling for clustering coefficient
sns.histplot(training_clustering, color='blue', kde=False, stat="density", bins=clustering_bins, ax=axes[0, 1], 
             line_kws={"linewidth": 2}, element="bars", alpha=0.7)
# Create custom x-ticks that show "0" for the first bin and nice spacing for others
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
    # Fix y-axis to reasonable density values
    ax.set_ylim(0, None)  # Let matplotlib determine upper bound based on data

# Ensure density scaling is correct across all plots
for row in axes:
    for ax in row:
        current_ylim = ax.get_ylim()
        if current_ylim[1] > 5:  # If scale appears to be percentage rather than density
            ax.set_ylim(0, min(5, current_ylim[1]/100))  # Cap at 5 for density scale

# Adjust layout and save the figure
plt.tight_layout(rect=[0.08, 0.04, 1, 0.97])  # Adjust to leave space for labels
plt.savefig('graph_statistics_comparison_columns.png', dpi=300, bbox_inches='tight')
# plt.show()

def plot_graph_comparison(training_samples, erdos_samples, vae_samples, save_path='graph_comparison.png', 
                         # Layout parameters
                         grid_height_ratios=[10, 10, 10, 1],
                         grid_hspace=0.3,
                         # Separator line positions (y-coordinates in figure space)
                         sep_line1_y=0.69,
                         sep_line2_y=0.36,
                         # Row label positions (y-coordinates in figure space)
                         train_label_y=0.83,
                         erdos_label_y=0.5,
                         vae_label_y=0.17,
                         # Layout adjustment parameters
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
        
    Layout Parameters:
    -----------------
    grid_height_ratios : list
        Height ratios for the rows in the GridSpec
    grid_hspace : float
        Horizontal spacing between subplots
    sep_line1_y : float
        Y-position of the first separator line (between training and Erdős-Rényi)
    sep_line2_y : float
        Y-position of the second separator line (between Erdős-Rényi and VAE)
    train_label_y : float
        Y-position of the "Training Graphs" label
    erdos_label_y : float
        Y-position of the "Erdős-Rényi Graphs" label
    vae_label_y : float
        Y-position of the "VAE Graphs" label
    tight_layout_rect : list
        Rectangle in which to fit the subplots [left, bottom, right, top]
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
    
    # Add horizontal separator line between training and Erdős-Rényi rows - adjusted position
    fig.add_artist(plt.Line2D([0.05, 0.95], [sep_line1_y, sep_line1_y], color='black', 
                             linewidth=1, transform=fig.transFigure))
    
    # Plot Erdos-Renyi graphs (middle row)
    for i, graph in enumerate(erdos_samples):
        ax = fig.add_subplot(gs[1, i])
        
        if isinstance(graph, torch.Tensor):
            graph = nx.from_numpy_array(graph.cpu().numpy())
            
        node_degrees = dict(graph.degree())
        node_colors = [node_degrees[n] for n in graph.nodes()]
        
        # Use spring layout with fixed parameters for more consistent sizing
        pos = nx.spring_layout(graph, seed=42)
        nx.draw(graph, pos=pos, ax=ax, node_size=40, with_labels=False,
                node_color=node_colors, arrows=False, cmap=cmap,
                vmin=0, vmax=global_max_degree)
        
        ax.set_title(f"Graph {i+1}", fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # Add horizontal separator line between Erdős-Rényi and VAE rows - adjusted position
    fig.add_artist(plt.Line2D([0.05, 0.95], [sep_line2_y, sep_line2_y], color='black', 
                             linewidth=1, transform=fig.transFigure))
    
    # Plot VAE graphs (bottom row)
    for i, graph in enumerate(vae_samples):
        ax = fig.add_subplot(gs[2, i])
        
        if isinstance(graph, torch.Tensor):
            graph = nx.from_numpy_array(graph.cpu().numpy())
            
        node_degrees = dict(graph.degree())
        node_colors = [node_degrees[n] for n in graph.nodes()]
        
        # Use spring layout with fixed parameters for more consistent sizing
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

# Example usage:
# To use this function, add something like this to your code:
# 
# # Get 3 random samples from each source
training_samples = random.sample(training_graphs, 3)
erdos_samples = random.sample(erdos_graphs, 3) 
vae_samples = random.sample(sampled_graphs, 3)

# Plot with default parameters
plot_graph_comparison(training_samples, erdos_samples, vae_samples, 'graph_model_comparison.png')

# Adjust parameters if needed (uncomment and modify as needed)
# plot_graph_comparison(training_samples, erdos_samples, vae_samples, 'graph_model_comparison_adjusted.png',
#                      sep_line1_y=0.71, sep_line2_y=0.38,  # Adjust separator lines
#                      train_label_y=0.85, erdos_label_y=0.52, vae_label_y=0.19)  # Adjust row labels
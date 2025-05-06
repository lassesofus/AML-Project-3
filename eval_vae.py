# %% 
import pdb
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree
from erdos_renyi import ErdosRenyiSampler
from architecture import get_vae
from utils import load_model_checkpoint


# Configs
device = 'cpu'
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
node_feature_dim = dataset.num_node_features  # should be 7
latent_dim = 16
batch_size = 16
model_path = './models/graph_vae.pt'


# Compute empirical distribution of the number of nodes in the dataset
num_nodes_list = [data.num_nodes for data in dataset]
# unique_node_counts, counts = torch.unique(torch.tensor(num_nodes_list), return_counts=True)
# node_count_distribution = counts.float() / counts.sum()  # Normalize to get probabilities
# max_nodes = unique_node_counts.max().item()  # Maximum number of nodes in the dataset


vae = get_vae(num_nodes_list=num_nodes_list)

vae = load_model_checkpoint(vae, model_path)

# Sample N graphs using the generator
N = 1000  # Number of graphs to sample

erdos = ErdosRenyiSampler(dataset)  # Adjust p as needed

def compute_metrics(sampled_graphs, training_graphs):
    sampled_graphs_nx = [nx.from_numpy_array(graph.cpu().numpy()) for graph in sampled_graphs]
    training_graphs_nx = [to_networkx(graph,to_undirected=True) for graph in training_graphs]
    # Novel
    novel_count = sum(
        not any(nx.is_isomorphic(g1, g2) for g2 in training_graphs_nx)
        for g1 in sampled_graphs_nx
    )
    novel_percentage = novel_count / len(sampled_graphs) * 100

    # Unique
    unique_graphs = [nx.weisfeiler_lehman_graph_hash(g) for g in sampled_graphs_nx]
    unique_count = len(set(unique_graphs))
    unique_percentage = unique_count / len(sampled_graphs) * 100

    # Novel and Unique
    exc = lambda s, i: s[:i] + s[i+1:]

    novel_and_unique_count = sum(
        not any(nx.is_isomorphic(g1, g2) for g2 in training_graphs_nx)
        and nx.weisfeiler_lehman_graph_hash(g1) not in
        exc(unique_graphs, i)
        for i,g1 in enumerate(sampled_graphs_nx)
    )
    novel_and_unique_percentage = novel_and_unique_count / len(sampled_graphs) * 100

    return novel_percentage, unique_percentage, novel_and_unique_percentage


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
novel, unique, novel_and_unique = compute_metrics(sampled_graphs, training_graphs)
print(f"Novel: {novel:.2f}%, Unique: {unique:.2f}%, Novel and Unique: {novel_and_unique:.2f}%")



import numpy as np
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

        # Compute eigenvector centrality with better error handling
        if graph_nx.number_of_nodes() > 0 and graph_nx.number_of_edges() > 0:
            # First try the default method
            ec_values = [0] * graph_nx.number_of_nodes()  # Default zeros
            
            # Method 1: Standard eigenvector centrality
            if nx.is_connected(graph_nx):
                # Only try eigenvector centrality if connected
                # Use power iteration method which is more robust
                try:
                    ec = nx.eigenvector_centrality_numpy(graph_nx, max_iter=1000)
                    ec_values = list(ec.values())
                except:
                    # Fall back to power iteration if numpy method fails
                    try:
                        ec = nx.eigenvector_centrality(graph_nx, max_iter=1000, tol=1e-3)
                        ec_values = list(ec.values())
                    except:
                        pass  # Keep zeros if both methods fail
            
            eigenvector_centrality_list.extend(ec_values)
        else:
            eigenvector_centrality_list.extend([0] * graph_nx.number_of_nodes())  # Handle empty graphs

    return degree_list, clustering_list, eigenvector_centrality_list


# Compute statistics for the training graphs
training_graphs = [data for data in dataset]
training_degrees, training_clustering, training_eigenvector = compute_graph_statistics(training_graphs)

# Compute statistics for the generated graphs
generated_degrees, generated_clustering, generated_eigenvector = compute_graph_statistics(sampled_graphs)
# Compute statistics for the Erdős-Rényi baseline
erdos_renyi_graphs = erdos.sample_graphs(N)
erdos_renyi_graphs = [g for g in erdos_renyi_graphs if g is not None]  # Filter out None values
erdos_renyi_degrees, erdos_renyi_clustering, erdos_renyi_eigenvector = compute_graph_statistics(erdos_renyi_graphs)
# Define the number of bins to ensure consistency across all histograms
num_bins = 30

# Create a 3-by-3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('Graph Statistics Comparison', fontsize=16)

# Compute global axis limits for each statistic
degree_min, degree_max = min(training_degrees + generated_degrees + erdos_renyi_degrees), max(training_degrees + generated_degrees + erdos_renyi_degrees)
clustering_min, clustering_max = min(training_clustering + generated_clustering + erdos_renyi_clustering), max(training_clustering + generated_clustering + erdos_renyi_clustering)
eigenvector_min, eigenvector_max = min(training_eigenvector + generated_eigenvector + erdos_renyi_eigenvector), max(training_eigenvector + generated_eigenvector + erdos_renyi_eigenvector)

# Row 1: Training Graphs
sns.histplot(training_degrees, color='blue', kde=False, stat="density", bins=num_bins, ax=axes[0, 0])
sns.histplot(training_clustering, color='blue', kde=False, stat="density", bins=num_bins, ax=axes[0, 1])
sns.histplot(training_eigenvector, color='blue', kde=False, stat="density", bins=num_bins, ax=axes[0, 2])
axes[0, 0].set_title('Node Degree')
axes[0, 1].set_title('Clustering Coefficient')
axes[0, 2].set_title('Eigenvector Centrality')
axes[0, 0].set_ylabel('Training Graphs')

# Row 2: Generated Graphs
sns.histplot(generated_degrees, color='green', kde=False, stat="density", bins=num_bins, ax=axes[1, 0])
sns.histplot(generated_clustering, color='green', kde=False, stat="density", bins=num_bins, ax=axes[1, 1])
sns.histplot(generated_eigenvector, color='green', kde=False, stat="density", bins=num_bins, ax=axes[1, 2])
axes[1, 0].set_ylabel('Generated Graphs')

# Row 3: Erdős-Rényi Baseline
sns.histplot(erdos_renyi_degrees, color='red', kde=False, stat="density", bins=num_bins, ax=axes[2, 0])
sns.histplot(erdos_renyi_clustering, color='red', kde=False, stat="density", bins=num_bins, ax=axes[2, 1])
sns.histplot(erdos_renyi_eigenvector, color='red', kde=False, stat="density", bins=num_bins, ax=axes[2, 2])
axes[2, 0].set_ylabel('Erdős-Rényi Baseline')

# Set consistent x-axis and y-axis limits for each column
for ax in [axes[0, 0], axes[1, 0], axes[2, 0]]:
    ax.set_xlim(degree_min, degree_max)
for ax in [axes[0, 1], axes[1, 1], axes[2, 1]]:
    ax.set_xlim(clustering_min, clustering_max)
for ax in [axes[0, 2], axes[1, 2], axes[2, 2]]:
    ax.set_xlim(eigenvector_min, eigenvector_max)

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title
plt.savefig('graph_statistics_comparison_columns.png')
plt.show()
import torch
import networkx as nx
from networkx.algorithms import weisfeiler_lehman_graph_hash
from tqdm import tqdm
import pdb
import random
import matplotlib.pyplot as plt
import numpy as np

def graph_to_nx(N, edge_index):
    """
    Convert edge_index and number of nodes into a NetworkX graph.
    Assumes edge_index is of shape [2, num_edges].
    """
    # If edge_index is a torch.Tensor, convert to list
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.tolist()
    edges = list(zip(edge_index[0], edge_index[1]))
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)
    return G

def compute_graph_hash(N, edge_index):
    """
    Compute the Weisfeiler-Lehman hash for a graph.
    """
    G = graph_to_nx(N, edge_index)
    return weisfeiler_lehman_graph_hash(G)

def compute_metrics(sample_hashes, training_hashes):
    """
    Given a list of sampled graph hashes and a set of training graph hashes,
    compute the percentage of:
      - Novel: graphs that do not appear in the training set.
      - Unique: unique graphs in the sample.
      - Novel and unique: unique graphs that are also novel.
    """
    total = len(sample_hashes)
    unique_hashes = set(sample_hashes)
    unique_percentage = 100 * len(unique_hashes) / total

    novel_hashes = [h for h in sample_hashes if h not in training_hashes]
    novel_percentage = 100 * len(novel_hashes) / total

    novel_unique_percentage = 100 * len(set(novel_hashes)) / total

    return novel_percentage, unique_percentage, novel_unique_percentage

def sample_and_evaluate(sampler, training_hashes, num_samples=1000):
    """
    Sample num_samples graphs using the provided sampler and
    return computed metrics (Novel, Unique, Novel+Unique).
    """
    sampled_hashes = []
    for _ in tqdm(range(num_samples)):
        N, r, edge_index, adj = sampler.sample()
        graph_hash = compute_graph_hash(N, edge_index)
        sampled_hashes.append(graph_hash)
    return compute_metrics(sampled_hashes, training_hashes)

def test_WL_hash_match():
    """
    Test if the WL hash is invariant to node relabeling.
    """
    # Create an example graph (e.g., a simple cycle)
    G = nx.cycle_graph(5)

    # Compute WL hash for the original graph
    hash_G = weisfeiler_lehman_graph_hash(G)
    print("Original Graph hash:", hash_G)

    # Create a permuted (transmuted) version of G:
    # Here we relabel the nodes randomly. The graphs remain isomorphic.
    nodes = list(G.nodes())
    permuted_nodes = nodes.copy()
    random.shuffle(permuted_nodes)
    mapping = {old: new for old, new in zip(nodes, permuted_nodes)}
    H = nx.relabel_nodes(G, mapping)

    # Compute WL hash for the permuted graph
    hash_H = weisfeiler_lehman_graph_hash(H)
    print("Permuted Graph hash:", hash_H)

    print("Hashes match:", hash_G == hash_H)

def test_WL_hash_non_isomorphic():
    """
    Test if the WL hash is different for non-isomorphic graphs.
    """
    # Create two non-isomorphic graphs
    G1 = nx.cycle_graph(5)  # Cycle graph with 5 nodes
    G2 = nx.path_graph(5)   # Path graph with 5 nodes

    # Compute WL hashes for both graphs
    hash_G1 = weisfeiler_lehman_graph_hash(G1)
    hash_G2 = weisfeiler_lehman_graph_hash(G2)

    print("Graph 1 hash:", hash_G1)
    print("Graph 2 hash:", hash_G2)

    print("Hashes match:", hash_G1 == hash_G2)

def permute_graph(G):
    """
    Randomly permute the nodes of a graph G. Shuffling the order of the nodes in the adjacency matrix
    does not change the graph structure, but it changes the node labels.
    """
    # Get the current node labels
    nodes = list(G.nodes())
    
    # Create a random permutation of the node labels
    permuted_nodes = nodes.copy()
    random.shuffle(permuted_nodes)
    
    # Create a mapping from old to new labels
    mapping = {old: new for old, new in zip(nodes, permuted_nodes)}
    
    # Relabel the graph using the mapping
    G_permuted = nx.relabel_nodes(G, mapping)
    
    return G_permuted

def get_graph_stats(graphs):
    """
    Compute node-level statistics for a list of graphs.
    Returns three lists:
      - degrees: Node degrees
      - clustering: Clustering coefficients
      - centrality: Eigenvector centrality values
    """
    degrees = []
    clustering = []
    centrality = []
    print("Computing node-level statistics...")
    for G in tqdm(graphs):
        # Append node degrees
        degrees.extend(list(dict(G.degree()).values()))
        
        # Append clustering coefficients for each node
        clustering.extend(list(nx.clustering(G).values()))
        
        # Append eigenvector centrality (if it cannot be computed, use 0 for each node)
        try:
            cent = nx.eigenvector_centrality_numpy(G)
            centrality.extend(list(cent.values()))
        except Exception:
            centrality.extend([0] * G.number_of_nodes())
    
    return (degrees, clustering, centrality)

def plot_histograms(baseline_stats, empirical_stats, deep_stats):

    baseline_degrees, baseline_clustering, baseline_centrality = baseline_stats
    empirical_degrees, empirical_clustering, empirical_centrality = empirical_stats
    deep_degrees, deep_clustering, deep_centrality = deep_stats

    # Compute common bins for each metric
    bins_degrees = np.linspace(min(baseline_degrees + empirical_degrees + deep_degrees), 
                                max(baseline_degrees + empirical_degrees + deep_degrees), 
                                30)
    bins_clustering = np.linspace(min(baseline_clustering + empirical_clustering + deep_clustering), 
                                max(baseline_clustering + empirical_clustering + deep_clustering), 
                                30)
    bins_centrality = np.linspace(min(baseline_centrality + empirical_centrality + deep_centrality), 
                                max(baseline_centrality + empirical_centrality + deep_centrality), 
                                30)

    # Create a 3x3 grid of subplots
    fig, ax = plt.subplots(3, 3, figsize=(15, 12))

    # Row 0: Empirical distributions (green)
    ax[0, 0].hist(empirical_degrees, bins=bins_degrees, alpha=0.5, label='Empirical', color='green')
    ax[0, 0].set_title('Degree Distribution', fontsize=14, fontweight='bold')
    ax[0, 0].set_ylabel('Frequency')

    ax[0, 1].hist(empirical_clustering, bins=bins_clustering, alpha=0.5, label='Empirical', color='green')
    ax[0, 1].set_title('Clustering Coefficient Distribution', fontsize=14, fontweight='bold')
    ax[0, 1].set_ylabel('Frequency')

    ax[0, 2].hist(empirical_centrality, bins=bins_centrality, alpha=0.5, label='Empirical', color='green')
    ax[0, 2].set_title('Centrality Distribution', fontsize=14, fontweight='bold')
    ax[0, 2].set_ylabel('Frequency')

    # Row 1: Baseline distributions (blue)
    ax[1, 0].hist(baseline_degrees, bins=bins_degrees, alpha=0.5, label='Baseline', color='blue')
    ax[1, 0].set_ylabel('Frequency')

    ax[1, 1].hist(baseline_clustering, bins=bins_clustering, alpha=0.5, label='Baseline', color='blue')
    ax[1, 1].set_ylabel('Frequency')

    ax[1, 2].hist(baseline_centrality, bins=bins_centrality, alpha=0.5, label='Baseline', color='blue')
    ax[1, 2].set_ylabel('Frequency')

    # Row 2: Deep generative model distributions (orange)
    ax[2, 0].hist(deep_degrees, bins=bins_degrees, alpha=0.5, label='Deep generative model', color='orange')
    ax[2, 0].set_ylabel('Frequency')

    ax[2, 1].hist(deep_clustering, bins=bins_clustering, alpha=0.5, label='Deep generative model', color='orange')
    ax[2, 1].set_ylabel('Frequency')

    ax[2, 2].hist(deep_centrality, bins=bins_centrality, alpha=0.5, label='Deep generative model', color='orange')
    ax[2, 2].set_ylabel('Frequency')

    # Add row labels by using fig.text
    fig.text(0.01, 0.85, "Empirical", va="center", ha="center", rotation="vertical",
             fontsize=14, fontweight="bold")
    fig.text(0.01, 0.5, "Baseline", va="center", ha="center", rotation="vertical",
             fontsize=14, fontweight="bold")
    fig.text(0.01, 0.175, "Deep generative model", va="center", ha="center", rotation="vertical",
             fontsize=14, fontweight="bold")

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.savefig('figures/graph_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_latex_table(results: dict) -> str:
    """
    Given a dictionary 'results' with keys 'baseline' and 'deep' and values as tuples:
    (novel_percentage, unique_percentage, novel_unique_percentage)
    this function returns a formatted LaTeX table as a string.
    """
    latex_table = r"""\begin{tabular}{lccc}
\hline
Model & Novel (\%) & Unique (\%) & Novel and Unique (\%) \\
\hline
Baseline & {baseline_novel:.2f} & {baseline_unique:.2f} & {baseline_novelunique:.2f} \\
Deep Generative Model & {deep_novel:.2f} & {deep_unique:.2f} & {deep_novelunique:.2f} \\
\hline
\end{tabular}""".format(
        baseline_novel=results['baseline'][0],
        baseline_unique=results['baseline'][1],
        baseline_novelunique=results['baseline'][2],
        deep_novel=results['deep'][0],
        deep_unique=results['deep'][1],
        deep_novelunique=results['deep'][2]
    )

def plot_graphs(graphs, title="Graphs"):
    """
    Plot a list of graphs using matplotlib in a 3x3 grid.
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()  # Flatten the 2D array into 1D for easier iteration.
    for i, G in enumerate(graphs[:9]):
        nx.draw(G, with_labels=True, ax=axes[i])
        axes[i].set_title(f"Graph {i + 1}")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"figures/{title}.png", dpi=300, bbox_inches='tight')
    plt.show()



class NodeDist():
    """
    A class to represent a node distribution.
    """
    def __init__(self, num_nodes_list):
        self.num_nodes_list = num_nodes_list
        self.unique_node_counts, counts = torch.unique(torch.tensor(num_nodes_list), return_counts=True)
        self.node_count_distribution = counts.float() / counts.sum()  # Normalize to get probabilities
        self.max_nodes = self.unique_node_counts.max().item()  # Maximum number of nodes in the dataset

        self.node_masks = []
        for i, n in enumerate(num_nodes_list):
            mask = torch.zeros(self.max_nodes, dtype=torch.bool)
            mask[:n] = True
            self.node_masks.append(mask)
        self.node_masks = torch.stack(self.node_masks)

    def sample(self, n_samples=1):
        indices = torch.multinomial(self.node_count_distribution, n_samples, replacement=True)
        return self.unique_node_counts[indices].tolist()
    
    def get_node_masks(self, nodes_per_graph):
        """
        Given a list of node counts, return the corresponding masks.
        """
        masks = []
        for n in nodes_per_graph:
            mask = torch.zeros((self.max_nodes,self.max_nodes), dtype=torch.bool)
            mask[:n,:n] = True
            mask.fill_diagonal_(False)
            masks.append(mask)
        return torch.stack(masks)

    
def load_model_checkpoint(model, checkpoint_path, device='cpu'):
    """
    Loads model and optimizer state from a checkpoint.

    Args:
        model (nn.Module): The model instance (must be already created with correct architecture).
        optimizer (torch.optim.Optimizer): The optimizer instance.
        checkpoint_path (str): Path to the .pt or .pth checkpoint file.
        device (str): 'cpu' or 'cuda' — where to load the model.

    Returns:
        model: The model with loaded weights.
        optimizer: The optimizer with loaded state.
        start_epoch: The epoch to resume from.
        loss: The best loss at the time of saving.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    start_epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)

    print(f"Loaded checkpoint from '{checkpoint_path}' (epoch {start_epoch}, loss {loss:.4f})")
    return model
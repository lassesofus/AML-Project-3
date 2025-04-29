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

def plot_histograms(baseline_stats, empirical_stats):

    baseline_degrees, baseline_clustering, baseline_centrality = baseline_stats
    empirical_degrees, empirical_clustering, empirical_centrality = empirical_stats

    # Compute common bins for each metric
    bins_degrees = np.linspace(min(baseline_degrees + empirical_degrees), 
                                max(baseline_degrees + empirical_degrees), 
                                30)
    bins_clustering = np.linspace(min(baseline_clustering + empirical_clustering), 
                                max(baseline_clustering + empirical_clustering), 
                                30)
    bins_centrality = np.linspace(min(baseline_centrality + empirical_centrality), 
                                max(baseline_centrality + empirical_centrality), 
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
    # ax[2, 0].hist(deep_degrees, bins=bins_degrees, alpha=0.5, label='Deep generative model', color='orange')
    # ax[2, 0].set_ylabel('Frequency')

    # ax[2, 1].hist(deep_clustering, bins=bins_clustering, alpha=0.5, label='Deep generative model', color='orange')
    # ax[2, 1].set_ylabel('Frequency')

    # ax[2, 2].hist(deep_centrality, bins=bins_centrality, alpha=0.5, label='Deep generative model', color='orange')
    # ax[2, 2].set_ylabel('Frequency')

    # Add row labels by using fig.text
    fig.text(0.01, 0.85, "Empirical", va="center", ha="center", rotation="vertical",
             fontsize=14, fontweight="bold")
    fig.text(0.01, 0.5, "Baseline", va="center", ha="center", rotation="vertical",
             fontsize=14, fontweight="bold")
    fig.text(0.01, 0.175, "Deep generative model", va="center", ha="center", rotation="vertical",
             fontsize=14, fontweight="bold")

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.savefig('graph_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()

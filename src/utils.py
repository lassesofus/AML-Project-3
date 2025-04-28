import torch
import networkx as nx
from networkx.algorithms import weisfeiler_lehman_graph_hash
from tqdm import tqdm
import pdb

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
    pdb.set_trace()
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
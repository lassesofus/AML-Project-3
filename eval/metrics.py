import torch
import networkx as nx
from torch_geometric.utils import to_networkx

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
    
    # Compute WL hashes for reference graphs (structure only)
    reference_hashes = set(nx.weisfeiler_lehman_graph_hash(g, node_attr=None) for g in reference_graphs_nx)

    # Compute WL hashes for candidate graphs
    candidate_hashes = [nx.weisfeiler_lehman_graph_hash(g) for g in candidate_graphs_nx]

    # A graph is novel if its hash is not in the reference set
    novel_list = [h not in reference_hashes for h in candidate_hashes]
    novel_count = sum(novel_list)
    novel_percentage = novel_count / len(candidate_graphs_nx) * 100

    # Compute uniqueness - Count distinct graph types
    graph_hashes = [nx.weisfeiler_lehman_graph_hash(g) for g in candidate_graphs_nx]
    unique_hashes = set(graph_hashes)  # Get unique hashes
    unique_count = len(unique_hashes)  # Number of distinct graph types
    unique_percentage = unique_count / len(candidate_graphs_nx) * 100
    
    # Count occurrences of each hash (for novel_and_unique calculation)
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

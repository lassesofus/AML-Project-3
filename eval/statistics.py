import torch
import networkx as nx
from torch_geometric.utils import to_networkx

def compute_graph_statistics(graphs):
    """
    Compute node degree, clustering coefficient, and eigenvector centrality for a list of graphs.
    
    Parameters:
    -----------
    graphs : list
        List of graphs to compute statistics for
        
    Returns:
    --------
    tuple: (degree_list, clustering_list, eigenvector_centrality_list)
    """
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

def print_clustering_stats(training_clustering, erdos_renyi_clustering, generated_clustering):
    """
    Print clustering coefficient distribution statistics.
    
    Parameters:
    -----------
    training_clustering : list
        Clustering coefficients from training graphs
    erdos_renyi_clustering : list
        Clustering coefficients from Erdős-Rényi graphs
    generated_clustering : list
        Clustering coefficients from VAE-generated graphs
    """
    print("Clustering coefficient distribution analysis:")
    print(f"Training - Zero values: {training_clustering.count(0)} out of {len(training_clustering)} "
          f"({100*training_clustering.count(0)/len(training_clustering):.2f}%)")
    print(f"Erdős-Rényi - Zero values: {erdos_renyi_clustering.count(0)} out of {len(erdos_renyi_clustering)} "
          f"({100*erdos_renyi_clustering.count(0)/len(erdos_renyi_clustering):.2f}%)")
    print(f"VAE - Zero values: {generated_clustering.count(0)} out of {len(generated_clustering)} "
          f"({100*generated_clustering.count(0)/len(generated_clustering):.2f}%)")

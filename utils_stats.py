import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

def calculate_degree_penalty(adj_matrices, num_nodes_per_graph=None, max_degree=3, progressive=True):
    """
    Compute a penalty based on how many nodes have degree > max_degree.
    """
    # Infer per-graph node counts if not provided
    B, N, _ = adj_matrices.shape
    if num_nodes_per_graph is None:
        num_nodes_per_graph = torch.full((B,), N, dtype=torch.long, device=adj_matrices.device)

    # Binarize adjacency for integer edge counts
    bin_adj = (adj_matrices > 0.5).float()
    # Compute degree per node
    deg = bin_adj.sum(dim=2)
    # Build mask to ignore padded nodes
    idx = torch.arange(N, device=adj_matrices.device).unsqueeze(0)
    mask = (idx < num_nodes_per_graph.unsqueeze(1)).float()
    
    # Compute excess degree over max_degree, only for real nodes
    if progressive:
        raw_excess = torch.clamp(deg - max_degree, min=0)
        excess = torch.pow(raw_excess, 2) * mask
    else:
        excess = torch.clamp(deg - max_degree, min=0) * mask
        
    # Normalize per graph by its real node count
    per_graph = excess.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return per_graph.mean()

def compute_graph_statistics(graphs):
    """
    Compute statistics for a list of graphs (supports various input formats).
    Returns: (degrees, clustering_coeffs, eigenvector_centralities)
    """
    degree_list = []
    clustering_list = []
    eigenvector_centrality_list = []
    
    for graph in graphs:
        # Convert to NetworkX graph based on input type
        if isinstance(graph, nx.Graph):
            graph_nx = graph
        elif isinstance(graph, torch.Tensor):
            graph_nx = nx.from_numpy_array(graph.cpu().numpy())
        elif isinstance(graph, dict) and 'edge_index' in graph:
            # Create a simple networkx graph from edge_index
            num_nodes = graph.get('num_nodes', 0)
            edge_index = graph['edge_index']
            
            # Convert edge_index to list of tuples
            if isinstance(edge_index, torch.Tensor):
                edges = [(edge_index[0, i].item(), edge_index[1, i].item()) 
                         for i in range(edge_index.size(1))]
            else:
                edges = edge_index
                
            graph_nx = nx.Graph()
            graph_nx.add_nodes_from(range(num_nodes))
            graph_nx.add_edges_from(edges)
        else:
            # Assume it's a PyG Data object
            try:
                graph_nx = to_networkx(graph, to_undirected=True)
            except:
                print(f"Could not convert to NetworkX graph: {type(graph)}")
                continue
            
        # Skip empty graphs
        if graph_nx.number_of_nodes() == 0:
            continue
            
        # Compute node degrees
        degrees = [d for _, d in graph_nx.degree()]
        degree_list.extend(degrees)
        
        # Compute clustering coefficients
        if graph_nx.number_of_edges() > 0:
            clustering = list(nx.clustering(graph_nx).values())
            clustering_list.extend(clustering)
        
        # Compute eigenvector centrality safely
        if graph_nx.number_of_edges() > 0:
            try:
                eigenvector_centrality = list(nx.eigenvector_centrality_numpy(graph_nx).values())
                eigenvector_centrality_list.extend(eigenvector_centrality)
            except:
                # Use zeros if calculation fails
                eigenvector_centrality_list.extend([0] * graph_nx.number_of_nodes())
        else:
            eigenvector_centrality_list.extend([0] * graph_nx.number_of_nodes())
        
    return degree_list, clustering_list, eigenvector_centrality_list

def calculate_active_units(model, dataloader, device, threshold=0.01):
    """Calculate the number of active units in the latent space."""
    model.eval()
    z_means = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            q = model.encoder(batch.x, batch.edge_index, batch.batch)
            z_mean = q.mean
            z_means.append(z_mean)
    
    # Concatenate all means
    all_means = torch.cat(z_means, dim=0)
    
    # Calculate variance for each dimension
    latent_variances = all_means.var(dim=0)
    
    # Count dimensions with variance above threshold
    active_dims = (latent_variances > threshold).sum().item()
    
    return active_dims, latent_variances.cpu().numpy()

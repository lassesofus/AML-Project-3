import torch
import networkx as nx

# Define device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_components(adj_matrix):
    """
    Find connected components in a graph
    
    Args:
        adj_matrix: Adjacency matrix [N, N]
        
    Returns:
        A list of component labels (same label = same component)
    """
    n = adj_matrix.size(0)
    component_labels = [-1] * n  # -1 means not visited
    current_component = 0
    
    for node in range(n):
        if component_labels[node] == -1:  # If not visited
            # Start a new component
            component_labels[node] = current_component
            
            # BFS to find all nodes in this component
            queue = [node]
            while queue:
                current = queue.pop(0)
                neighbors = torch.where(adj_matrix[current] > 0.5)[0].tolist()
                
                for neighbor in neighbors:
                    if component_labels[neighbor] == -1:
                        component_labels[neighbor] = current_component
                        queue.append(neighbor)
            
            current_component += 1
    
    return component_labels

def count_connected_components(adj_matrix, node_mask=None):
    """
    Count connected components in a graph
    
    Args:
        adj_matrix: Adjacency matrix [N, N]
        node_mask: Binary mask [N] indicating real nodes
        
    Returns:
        Number of connected components
    """
    adj_binary = (adj_matrix > 0.5).float()
    
    if node_mask is not None:
        mask_2d = node_mask.unsqueeze(1) * node_mask.unsqueeze(0)
        adj_binary = adj_binary * mask_2d
        
        # Create a mask for real nodes
        node_mask_bool = node_mask.bool()
        n_real_nodes = node_mask_bool.sum().item()
        if n_real_nodes == 0:
            return 0
    else:
        node_mask_bool = torch.ones(adj_binary.size(0), dtype=torch.bool, device=adj_binary.device)
    
    # Get component labels for real nodes
    component_labels = find_components(adj_binary)
    
    # Count unique component ids among real nodes
    real_component_labels = [component_labels[i] for i in range(len(component_labels)) if node_mask_bool[i]]
    unique_components = set(real_component_labels)
    
    return len(unique_components)

def calculate_degree_metrics(adj_probs, node_mask=None):
    """
    Calculate degree-related metrics from probabilistic adjacency matrix
    
    Args:
        adj_probs: Probabilistic adjacency matrix [B, N, N]
        node_mask: Binary mask [B, N] indicating real nodes
        
    Returns:
        degrees: Average degree per node [B, N]
        degree_dist: Distribution of degrees [B, 7] (0-5, 6+)
    """
    batch_size, n_nodes, _ = adj_probs.shape
    
    if node_mask is None:
        node_mask = torch.ones(batch_size, n_nodes, device=adj_probs.device)
    
    # Calculate degrees (sum of edge probabilities)
    mask_2d = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)  # [B, N, N]
    degrees = (adj_probs * mask_2d).sum(dim=-1)  # [B, N]
    
    # Calculate degree distribution
    degree_dist = torch.zeros(batch_size, 7, device=device)  # [B, 7]
    
    for b in range(batch_size):
        real_degrees = degrees[b, node_mask[b].bool()]
        n_real_nodes = real_degrees.size(0)
        
        if n_real_nodes == 0:
            continue
        
        # Count degrees
        for d in range(6):  # 0, 1, 2, 3, 4, 5
            degree_dist[b, d] = torch.sum((real_degrees >= d) & (real_degrees < d + 1)) / n_real_nodes
        
        # Count degrees 6+
        degree_dist[b, 6] = torch.sum(real_degrees >= 6) / n_real_nodes
    
    return degrees, degree_dist

def pyg_to_networkx(data):
    """
    Convert a PyTorch Geometric data object to a NetworkX graph
    
    Args:
        data: PyG Data object
        
    Returns:
        G: NetworkX Graph
    """
    G = nx.Graph()
    
    # Add nodes
    for i in range(data.num_nodes):
        G.add_node(i)
    
    # Add edges
    edge_list = data.edge_index.t().tolist()
    G.add_edges_from(edge_list)
    
    return G
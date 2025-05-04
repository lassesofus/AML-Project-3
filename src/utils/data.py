import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(root='./data/', name='MUTAG'):
    """
    Load graph dataset and analyze its properties
    
    Args:
        root: Path to data directory
        name: Name of dataset (e.g., 'MUTAG')
    
    Returns:
        dataset: PyTorch Geometric dataset
        node_feature_dim: Dimension of node features
        max_nodes: Maximum number of nodes across all graphs
        target_dist: Target degree distribution [7] (0-5, 6+)
        node_counts: List of node counts for each graph
    """
    dataset = TUDataset(root=root, name=name).to(device)
    node_feature_dim = dataset.num_node_features
    
    # Get max nodes and analyze degree distribution in original dataset
    max_nodes = max([data.num_nodes for data in dataset])
    
    # Track node count distribution
    node_counts = [data.num_nodes for data in dataset]
    mean_nodes = sum(node_counts) / len(node_counts)
    
    # Calculate average degrees in original data
    all_degrees = []
    degree_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}  # Initialize counts
    
    for data in dataset:
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        degrees = torch.zeros(num_nodes, device=device)
        
        # Count degree for each node
        for i in range(num_nodes):
            degrees[i] = torch.sum(edge_index[0] == i) + torch.sum(edge_index[1] == i)
        
        all_degrees.extend(degrees.tolist())
        
        # Count occurrences of each degree
        for d in degrees:
            d_int = int(d.item())
            if d_int <= 5:
                degree_counts[d_int] += 1
            else:
                degree_counts[6] += 1  # Group all degrees > 5
    
    # Convert to target distribution probabilities (frequencies)
    total_nodes = len(all_degrees)
    target_dist = torch.zeros(7, device=device)  # [deg 0, 1, 2, 3, 4, 5, 6+]
    for i in range(7):
        count = degree_counts[i] if i < 6 else degree_counts[6]
        target_dist[i] = count / total_nodes
    
    avg_degree = sum(all_degrees) / len(all_degrees)
    print(f"Original dataset - Avg degree: {avg_degree:.2f}")
    print(f"Original degree distribution: {target_dist}")
    print(f"Original dataset - Avg nodes: {mean_nodes:.2f}, Min: {min(node_counts)}, Max: {max_nodes}")
    
    return dataset, node_feature_dim, max_nodes, target_dist, node_counts

def create_dataloaders(dataset, batch_size=32, seed=0):
    """
    Split dataset into train/validation/test and create dataloaders
    
    Args:
        dataset: PyTorch Geometric dataset
        batch_size: Batch size for dataloaders
        seed: Random seed for reproducibility
        
    Returns:
        train_loader: DataLoader for training set
        validation_loader: DataLoader for validation set
        test_loader: DataLoader for test set
    """
    rng = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, validation_loader, test_loader

def prepare_batch_targets(data, max_nodes, node_feature_dim, device):
    """
    Prepare padded tensors and masks from a batch of PyG data
    
    Args:
        data: PyG Data batch
        max_nodes: Maximum number of nodes to pad to
        node_feature_dim: Node feature dimension
        device: Device to create tensors on
        
    Returns:
        adj_target: Padded adjacency matrix [B, N, N]
        padded_x: Padded node features [B, N, F]
        node_mask: Binary mask for real nodes [B, N]
        node_counts: List of node counts per graph
    """
    batch_size = data.batch.max().item() + 1
    adj_target = to_dense_adj(data.edge_index, data.batch, max_num_nodes=max_nodes)
    
    padded_x = torch.zeros(batch_size, max_nodes, node_feature_dim, device=device)
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.float32, device=device)
    
    node_counts = []
    for i in range(batch_size):
        indices = (data.batch == i).nonzero(as_tuple=True)[0]
        nodes_in_graph = len(indices)
        node_counts.append(nodes_in_graph)
        
        if nodes_in_graph > 0:
            current_x = data.x[indices]
            padded_x[i, :nodes_in_graph] = current_x
            node_mask[i, :nodes_in_graph] = 1.0
    
    return adj_target, padded_x, node_mask, node_counts
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random  # Added missing import

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper function for interactive plots
def drawnow():
    """Force draw the current plot."""
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()

# Load the MUTAG dataset
def load_data(root='./data/', name='MUTAG'):
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
    print(f"Original dataset - Avg nodes: {mean_nodes:.2f}, Min: {min(node_counts)}, Max: {max(node_counts)}")
    
    return dataset, node_feature_dim, max_nodes, target_dist, node_counts

# Split data and create dataloaders
def create_dataloaders(dataset, batch_size=32, seed=0):
    rng = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, validation_loader, test_loader

# Graph Message Passing Layer
class GraphMessagePassing(nn.Module):
    """Simple message passing layer for graph neural networks"""
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.message_net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )
        self.update_net = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )
    
    def forward(self, x, edge_index):
        # Compute messages
        messages = self.message_net(x)
        
        # Aggregate messages (sum)
        num_nodes = x.size(0)
        aggregated = x.new_zeros((num_nodes, messages.size(1)))
        edge_index = edge_index.long()
        aggregated = aggregated.index_add(0, edge_index[1], messages[edge_index[0]])
        
        # Update node states
        updated = self.update_net(aggregated)
        
        return updated + x  # Residual connection

# Graph Encoder
class GraphEncoder(nn.Module):
    """Encoder for Graph VAE with graph-level latent"""
    
    def __init__(self, node_feature_dim, hidden_dim, latent_dim, num_layers=3):
        super().__init__()
        
        # Initial feature projection
        self.input_net = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            GraphMessagePassing(hidden_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # Graph-level pooling output projections
        self.mean_net = nn.Linear(hidden_dim, latent_dim)
        self.logvar_net = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x, edge_index, batch):
        # Initial projection
        h = self.input_net(x)
        
        # Message passing
        for layer in self.message_layers:
            h = layer(h, edge_index)
        
        # Graph-level pooling (sum)
        num_graphs = batch.max().item() + 1
        graph_embedding = torch.zeros(num_graphs, h.size(1), device=h.device)
        graph_embedding = graph_embedding.index_add(0, batch, h)
        
        # Project to latent distribution parameters
        z_mean = self.mean_net(graph_embedding)
        z_logvar = self.logvar_net(graph_embedding)
        
        return z_mean, z_logvar

# Graph Decoder
class GraphDecoder(nn.Module):
    """Decoder for Graph VAE to reconstruct graphs from latent variables"""
    
    def __init__(self, latent_dim, hidden_dim, max_nodes=28, node_feature_dim=7, target_degree_dist=None):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim
        self.target_degree_dist = target_degree_dist
        
        # Latent to initial graph representation
        self.latent_to_graph = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Generate node embeddings from graph representation
        self.node_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * max_nodes),
            nn.ReLU()
        )
        
        # Predict node features
        self.node_features = nn.Linear(hidden_dim, node_feature_dim)
        
        # Predict adjacency matrix with additional constraints to reduce density
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            # No activation - we apply sigmoid later
        )
    
    def forward(self, z, node_counts=None):
        batch_size = z.size(0)
        
        # Get graph representation from latent
        graph_repr = self.latent_to_graph(z)
        
        # Generate node embeddings for each graph
        all_node_embeds = self.node_embedding(graph_repr)
        all_node_embeds = all_node_embeds.view(batch_size, self.max_nodes, -1)
        
        # Predict node features
        node_features = self.node_features(all_node_embeds)
        
        # Create pairs of node embeddings for edge prediction
        nodes_i = all_node_embeds.unsqueeze(2).repeat(1, 1, self.max_nodes, 1)
        nodes_j = all_node_embeds.unsqueeze(1).repeat(1, self.max_nodes, 1, 1)
        
        # Concatenate node embeddings for each pair
        edge_inputs = torch.cat([nodes_i, nodes_j], dim=-1)
        
        # Reshape for the edge predictor
        edge_inputs = edge_inputs.view(batch_size, self.max_nodes * self.max_nodes, -1)
        
        # Predict edges
        edge_logits = self.edge_predictor(edge_inputs).view(batch_size, self.max_nodes, self.max_nodes)
        
        # Make adjacency matrix symmetric (undirected graph)
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2
        
        # Set diagonal to very negative values (no self-loops)
        mask = torch.eye(self.max_nodes, device=edge_logits.device).unsqueeze(0)
        edge_logits = edge_logits * (1 - mask) - 1e9 * mask
        
        return node_features, edge_logits

# Complete Graph VAE
class GraphVAE(nn.Module):
    """Variational Autoencoder for Graphs with graph-level latent variable"""
    
    def __init__(self, node_feature_dim, hidden_dim, latent_dim, max_nodes=28, num_layers=3, target_degree_dist=None):
        super().__init__()
        
        self.encoder = GraphEncoder(node_feature_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = GraphDecoder(latent_dim, hidden_dim, max_nodes, node_feature_dim, target_degree_dist)
        self.latent_dim = latent_dim
        self.max_nodes = max_nodes
        self.target_degree_dist = target_degree_dist
    
    def encode(self, x, edge_index, batch):
        return self.encoder(x, edge_index, batch)
    
    def decode(self, z, node_counts=None):
        return self.decoder(z, node_counts)
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick for sampling from latent distribution"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x, edge_index, batch, node_counts=None):
        # Encode
        z_mean, z_logvar = self.encode(x, edge_index, batch)
        
        # Sample latent variable
        z = self.reparameterize(z_mean, z_logvar)
        
        # Decode
        node_features, adj_logits = self.decode(z, node_counts)
        
        return node_features, adj_logits, z_mean, z_logvar
    
    def sample(self, num_samples=1):
        """
        Sample new graphs from the prior with enforced connectivity and controlled degree distribution
        
        Args:
            num_samples: Number of graphs to sample
        """
        # Sample from the latent prior
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Decode to get node features and edge logits
        node_features, adj_logits = self.decode(z)
        
        # Convert logits to probabilities
        adj_probs = torch.sigmoid(adj_logits)
        
        # Prepare output structures
        batch_size = z.size(0)
        sampled_adj = torch.zeros_like(adj_probs)
        
        # Ideal degree distribution (from MUTAG dataset)
        # We want about 20% degree 1, 40% degree 2, 40% degree 3
        target_degree_percentages = [0.0, 0.20, 0.40, 0.40, 0.0, 0.0, 0.0]  # [deg 0, 1, 2, 3, 4, 5, 6+]
        
        for b in range(batch_size):
            # Create target degree distribution for this graph
            n_nodes = self.max_nodes
            target_degrees = []
            for degree, percentage in enumerate(target_degree_percentages):
                if degree > 0:  # Skip degree 0
                    num_nodes_with_degree = int(n_nodes * percentage)
                    target_degrees.extend([degree] * num_nodes_with_degree)
            
            # If we don't have enough degrees due to rounding, add more deg 2-3 nodes
            while len(target_degrees) < n_nodes:
                target_degrees.append(random.choice([2, 3]))
                
            # Shuffle target degrees
            random.shuffle(target_degrees)
            
            # Create a list of potential edges sorted by probability
            edge_list = []
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):  # Upper triangle only
                    edge_list.append((i, j, adj_probs[b, i, j].item()))
            
            # Sort by probability (highest first)
            edge_list.sort(key=lambda x: x[2], reverse=True)
            
            # Track current node degrees
            degrees = torch.zeros(n_nodes, device=device)
            
            # First create a spanning tree to ensure connectivity
            # Use Union-Find data structure
            parents = list(range(n_nodes))
            
            def find(x):
                if parents[x] != x:
                    parents[x] = find(parents[x])
                return parents[x]
            
            def union(x, y):
                parents[find(x)] = find(y)
            
            # Initialize with one edge to start the spanning tree
            for i, j, prob in edge_list:
                # Check if we can add this edge based on target degrees
                if degrees[i] < target_degrees[i] and degrees[j] < target_degrees[j]:
                    sampled_adj[b, i, j] = 1.0
                    sampled_adj[b, j, i] = 1.0
                    degrees[i] += 1
                    degrees[j] += 1
                    union(i, j)
                    break
            
            # Continue building spanning tree (connect all components)
            edges_needed = n_nodes - 1  # Spanning tree needs n-1 edges
            edges_added = 1
            
            # Track components
            components = {}
            for e_idx, (i, j, prob) in enumerate(edge_list):
                # Skip if already added
                if sampled_adj[b, i, j] > 0:
                    continue
                    
                # Only add edge if it connects two separate components
                if find(i) != find(j):
                    # Check if we can add this edge based on target degrees
                    if degrees[i] < target_degrees[i] and degrees[j] < target_degrees[j]:
                        sampled_adj[b, i, j] = 1.0
                        sampled_adj[b, j, i] = 1.0
                        degrees[i] += 1
                        degrees[j] += 1
                        union(i, j)
                        edges_added += 1
                    
                # Ensure we have enough edges for a spanning tree
                if edges_added >= edges_needed:
                    # Check if graph is fully connected
                    root_id = find(0)
                    all_connected = all(find(node) == root_id for node in range(n_nodes))
                    if all_connected:
                        break
            
            # Count components to check connectivity
            unique_components = set()
            for node in range(n_nodes):
                unique_components.add(find(node))
            
            # If still multiple components, force connectivity while respecting degree constraints
            if len(unique_components) > 1:
                # Identify components
                component_to_nodes = {}
                for node in range(n_nodes):
                    comp = find(node)
                    if comp not in component_to_nodes:
                        component_to_nodes[comp] = []
                    component_to_nodes[comp].append(node)
                
                # Connect components
                components_list = list(component_to_nodes.keys())
                for i in range(1, len(components_list)):
                    # Try to find nodes that can be connected
                    found_connection = False
                    for node1 in component_to_nodes[components_list[0]]:
                        if degrees[node1] >= target_degrees[node1]:
                            continue
                        for node2 in component_to_nodes[components_list[i]]:
                            if degrees[node2] >= target_degrees[node2]:
                                continue
                                
                            # Connect these components
                            sampled_adj[b, node1, node2] = 1.0
                            sampled_adj[b, node2, node1] = 1.0
                            degrees[node1] += 1
                            degrees[node2] += 1
                            union(node1, node2)
                            found_connection = True
                            break
                        if found_connection:
                            break
                    
                    # If no connection possible with degree constraints, relax constraint for one node
                    if not found_connection:
                        node1 = component_to_nodes[components_list[0]][0]
                        node2 = component_to_nodes[components_list[i]][0]
                        sampled_adj[b, node1, node2] = 1.0
                        sampled_adj[b, node2, node1] = 1.0
                        degrees[node1] += 1
                        degrees[node2] += 1
                        union(node1, node2)
            
            # Add additional edges to match target degrees, but respect target constraints
            for i in range(n_nodes):
                # Skip if already at or above target degree
                if degrees[i] >= target_degrees[i]:
                    continue
                
                # Find potential neighbors for this node
                potential_neighbors = []
                for j in range(n_nodes):
                    if i != j and sampled_adj[b, i, j] == 0 and degrees[j] < target_degrees[j]:
                        potential_neighbors.append((j, adj_probs[b, i, j].item()))
                
                # Sort by probability
                potential_neighbors.sort(key=lambda x: x[1], reverse=True)
                
                # Add edges until reaching target degree
                for j, prob in potential_neighbors:
                    if degrees[i] >= target_degrees[i] or degrees[j] >= target_degrees[j]:
                        continue
                        
                    # Add edge
                    sampled_adj[b, i, j] = 1.0
                    sampled_adj[b, j, i] = 1.0
                    degrees[i] += 1
                    degrees[j] += 1
                    
                    if degrees[i] >= target_degrees[i]:
                        break
        
        return node_features, sampled_adj

    def interpolate(self, x1, edge_index1, batch1, x2, edge_index2, batch2, steps=5):
        """Interpolate between two graphs in latent space"""
        # Encode both graphs
        z_mean1, _ = self.encode(x1, edge_index1, batch1)
        z_mean2, _ = self.encode(x2, edge_index2, batch2)
        
        # Interpolate in latent space
        alphas = torch.linspace(0, 1, steps, device=device)
        interpolations = []
        
        for alpha in alphas:
            z_interp = z_mean1 * (1 - alpha) + z_mean2 * alpha
            node_features, adj_logits = self.decode(z_interp)
            interpolations.append((node_features, adj_logits))
        
        return interpolations

# Find connected components in a graph
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

# Count connected components
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

# Calculate degree metrics and distribution
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

# Prepare targets and masks for a batch
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
    start_idx = 0
    for i in range(batch_size):
        indices = (data.batch == i).nonzero(as_tuple=True)[0]
        nodes_in_graph = len(indices)
        node_counts.append(nodes_in_graph)
        
        if nodes_in_graph > 0:
            current_x = data.x[indices]
            padded_x[i, :nodes_in_graph] = current_x
            node_mask[i, :nodes_in_graph] = 1.0
    
    return adj_target, padded_x, node_mask, node_counts

# Calculate VAE loss with improved structure penalties
def calculate_vae_loss(adj_logits, node_features, adj_target, node_target, node_mask,
                      z_mean, z_logvar, beta, target_degree_dist, config):
    """
    Calculate complete VAE loss with structural penalties
    
    Args:
        adj_logits: Predicted adjacency logits [B, N, N]
        node_features: Predicted node features [B, N, F]
        adj_target: Target adjacency matrix [B, N, N]
        node_target: Target node features [B, N, F]
        node_mask: Binary mask for real nodes [B, N]
        z_mean: Latent mean [B, Z]
        z_logvar: Latent log-variance [B, Z]
        beta: KL divergence weight
        target_degree_dist: Target degree distribution [7]
        config: Dictionary of penalty weights and parameters
        
    Returns:
        Dictionary of loss components
    """
    batch_size, max_nodes, _ = node_features.shape
    epsilon = 1e-8
    
    # === Base VAE losses ===
    # Adjacency reconstruction loss
    mask_2d = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
    adj_recon_loss = F.binary_cross_entropy_with_logits(
        adj_logits, adj_target, reduction='none'
    )
    adj_recon_loss = (adj_recon_loss * mask_2d).sum() / (mask_2d.sum() + epsilon)
    
    # Node feature reconstruction loss
    node_mask_flat = node_mask.view(-1, 1)
    node_pred_flat = node_features.view(-1, node_features.size(-1))
    node_target_flat = node_target.view(-1, node_target.size(-1))
    
    node_recon_loss = F.mse_loss(
        node_pred_flat * node_mask_flat,
        node_target_flat * node_mask_flat,
        reduction='sum'
    ) / (node_mask_flat.sum() + epsilon)
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp()) / batch_size
    
    # === Graph structure penalties ===
    adj_probs = torch.sigmoid(adj_logits)
    
    # Get degree metrics
    degrees, pred_degree_dist = calculate_degree_metrics(adj_probs, node_mask)
    avg_pred_dist = pred_degree_dist.mean(dim=0)  # Average over batch
    
    # 1. Degree distribution matching loss (L1 distance)
    degree_dist_loss = F.l1_loss(avg_pred_dist, target_degree_dist)
    
    # 2. Valency penalty (penalize degrees > max_degree)
    max_degree = config['max_degree']
    degrees_masked = degrees * node_mask
    penalty_valency = torch.relu(degrees_masked - max_degree).sum() / (node_mask.sum() + epsilon)
    
    # 3. Edge sparsity regularization
    # This penalties the total number of edges to encourage sparser graphs
    edge_density = (adj_probs * mask_2d).sum() / (mask_2d.sum() + epsilon)
    target_density = config['target_edge_density']  # Set based on original dataset
    sparsity_loss = F.mse_loss(edge_density, torch.tensor(target_density).to(device))
    
    # 4. Connectivity loss
    connectivity_loss = torch.tensor(0.0, device=device)
    for b in range(batch_size):
        graph_adj = adj_probs[b]
        graph_mask = node_mask[b]
        
        if graph_mask.sum() > 1:  # Only check if more than 1 node
            n_components = count_connected_components(graph_adj, graph_mask)
            # Strongly penalize more than 1 component
            connectivity_loss += torch.relu(torch.tensor(n_components - 1.0, device=device))
    
    connectivity_loss = connectivity_loss / batch_size
    
    # === Combine all losses ===
    # Weight each penalty term
    weighted_degree_dist = config['distribution_weight'] * degree_dist_loss
    weighted_valency = config['valency_weight'] * penalty_valency
    weighted_sparsity = config['sparsity_weight'] * sparsity_loss
    weighted_connectivity = config['connectivity_weight'] * connectivity_loss
    
    # Combined structure penalty
    structure_penalty = (
        weighted_degree_dist +
        weighted_valency +
        weighted_sparsity +
        weighted_connectivity
    )
    
    # Total loss
    recon_loss = adj_recon_loss + node_recon_loss
    total_loss = recon_loss + beta * kl_loss + structure_penalty
    
    return {
        'total': total_loss,
        'reconstruction': recon_loss, 
        'kl': kl_loss,
        'structure_penalty': structure_penalty,
        'degree_dist': degree_dist_loss,
        'valency': penalty_valency,
        'sparsity': sparsity_loss,
        'connectivity': connectivity_loss,
        'pred_degree_dist': avg_pred_dist.detach(),
        'target_degree_dist': target_degree_dist
    }

# Plot training curves
def plot_curves(train_history, val_history, epoch):
    # Create figure for loss curves
    plt.figure('Loss', figsize=(12, 5))
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(train_history['total'], label='Train Total Loss')
    plt.plot(val_history['total'], label='Val Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.title('Total Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_history['reconstruction'], label='Recon')
    plt.plot(train_history['kl'], label='KL')
    plt.plot(train_history['structure_penalty'], label='Structure Penalty')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Component')
    plt.legend()
    plt.yscale('log')
    plt.title('Loss Components')
    plt.tight_layout()
    drawnow()
    
    # Create figure for penalties
    plt.figure('Penalties', figsize=(12, 5))
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(train_history['degree_dist'], label='Degree Dist')
    plt.plot(train_history['connectivity'], label='Connectivity')
    plt.xlabel('Epoch')
    plt.ylabel('Penalty Value')
    plt.legend()
    plt.title('Distribution & Connectivity Penalties')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_history['valency'], label='Valency')
    plt.plot(train_history['sparsity'], label='Sparsity')
    plt.xlabel('Epoch')
    plt.ylabel('Penalty Value')
    plt.legend()
    plt.title('Valency & Sparsity Penalties')
    plt.tight_layout()
    drawnow()
    
    # Create figure for disconnected graph ratio
    plt.figure('Stats', figsize=(12, 5))
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(val_history['disconnected_ratio'], label='Disconnected')
    plt.ylabel('Ratio')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.title('Disconnected Graph Ratio')
    
    # Plot latest degree distribution comparison
    plt.subplot(1, 2, 2)
    width = 0.35
    x = np.arange(7)
    
    target_dist = val_history['target_degree_dist'][-1].cpu().numpy()
    pred_dist = val_history['pred_degree_dist'][-1].cpu().numpy()
    
    plt.bar(x - width/2, target_dist, width, label='Target')
    plt.bar(x + width/2, pred_dist, width, label='Predicted')
    plt.xticks(x, ['0', '1', '2', '3', '4', '5', '6+'])
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Degree Distribution')
    plt.tight_layout()
    drawnow()

# Evaluate the model
def evaluate_model(model, loader, max_nodes, node_feature_dim, beta, target_degree_dist, config, device):
    model.eval()
    total_results = {
        'total': 0,
        'reconstruction': 0,
        'kl': 0,
        'structure_penalty': 0,
        'degree_dist': 0,
        'valency': 0,
        'sparsity': 0,
        'connectivity': 0,
        'disconnected_graphs': 0
    }
    total_batches = 0
    total_graphs = 0
    pred_degree_dist_sum = torch.zeros_like(target_degree_dist)
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Prepare batch data
            adj_target, node_target, node_mask, _ = prepare_batch_targets(
                data, max_nodes, node_feature_dim, device
            )
            batch_size = adj_target.size(0)
            total_graphs += batch_size
            
            # Forward pass
            node_features, adj_logits, z_mean, z_logvar = model(
                data.x, data.edge_index, data.batch
            )
            
            # Compute loss components
            loss_dict = calculate_vae_loss(
                adj_logits, node_features, adj_target, node_target, node_mask,
                z_mean, z_logvar, beta, target_degree_dist, config
            )
            
            # Update metrics
            for k in total_results.keys():
                if k in loss_dict:
                    total_results[k] += loss_dict[k].item() * batch_size
            
            # Sum predicted degree distributions
            pred_degree_dist_sum += loss_dict['pred_degree_dist'] * batch_size
            
            # Count disconnected graphs
            adj_probs = torch.sigmoid(adj_logits)
            for b in range(batch_size):
                graph_adj = adj_probs[b]
                graph_mask = node_mask[b]
                if graph_mask.sum() > 1:  # Skip empty graphs
                    n_components = count_connected_components(graph_adj, graph_mask)
                    if n_components > 1:
                        total_results['disconnected_graphs'] += 1
            
            total_batches += 1
    
    # Calculate averages
    results = {k: v / total_graphs for k, v in total_results.items()}
    results['disconnected_ratio'] = total_results['disconnected_graphs'] / total_graphs
    results['pred_degree_dist'] = pred_degree_dist_sum / total_graphs
    results['target_degree_dist'] = target_degree_dist
    
    return results

# KL annealing scheduler
def kl_annealing_factor(epoch, config):
    """Calculate KL annealing factor based on current epoch"""
    start = config['kl_annealing_start']
    end = config['kl_annealing_end']
    final_beta = config['final_beta']
    
    if epoch < start:
        return 0.0
    elif epoch >= end:
        return final_beta
    else:
        # Linear annealing
        return final_beta * (epoch - start) / (end - start)

# Train the model
def train_graph_vae(config):
    """Train the Graph VAE model with the specified configuration"""
    # === Setup ===
    hidden_dim = config['hidden_dim']
    latent_dim = config['latent_dim']
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    num_layers = config['num_layers']
    model_save_path = config['model_save_path']
    target_degree_dist = config['target_degree_dist'].to(device)
    
    # Load dataset and create dataloaders
    dataset, node_feature_dim, max_nodes, _, _ = load_data()
    train_loader, validation_loader, _ = create_dataloaders(dataset, batch_size)
    
    # Initialize model and optimizer
    model = GraphVAE(
        node_feature_dim, hidden_dim, latent_dim, max_nodes, num_layers, target_degree_dist
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize history trackers
    train_history = {
        'total': [], 'reconstruction': [], 'kl': [], 'structure_penalty': [],
        'degree_dist': [], 'valency': [], 'sparsity': [], 'connectivity': [],
        'pred_degree_dist': [], 'target_degree_dist': []
    }
    
    val_history = {
        'total': [], 'reconstruction': [], 'kl': [], 'structure_penalty': [],
        'degree_dist': [], 'valency': [], 'sparsity': [], 'connectivity': [],
        'disconnected_ratio': [], 'pred_degree_dist': [], 'target_degree_dist': []
    }
    
    # Setup plotting
    plt.ion()
    
    # === Training Loop ===
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {
            'total': 0.0, 'reconstruction': 0.0, 'kl': 0.0, 'structure_penalty': 0.0,
            'degree_dist': 0.0, 'valency': 0.0, 'sparsity': 0.0, 'connectivity': 0.0
        }
        pred_degree_dist_sum = torch.zeros_like(target_degree_dist)
        num_batches = 0
        
        # Calculate current KL annealing factor
        current_beta = kl_annealing_factor(epoch, config)
        
        for data in train_loader:
            data = data.to(device)
            batch_size = data.batch.max().item() + 1
            
            # Prepare batch data
            adj_target, node_target, node_mask, _ = prepare_batch_targets(
                data, max_nodes, node_feature_dim, device
            )
            
            # Forward pass
            node_features, adj_logits, z_mean, z_logvar = model(
                data.x, data.edge_index, data.batch
            )
            
            # Compute loss
            loss_dict = calculate_vae_loss(
                adj_logits, node_features, adj_target, node_target, node_mask,
                z_mean, z_logvar, current_beta, target_degree_dist, config
            )
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()
            
            # Update metrics (only scalar values)
            for k in epoch_losses.keys():
                epoch_losses[k] += loss_dict[k].item()
            
            # Handle tensor metrics separately
            pred_degree_dist_sum += loss_dict['pred_degree_dist']
            
            num_batches += 1
        
        # Record training metrics
        for k, v in epoch_losses.items():
            train_history[k].append(v / num_batches)
        
        # Record tensor metrics
        train_history['pred_degree_dist'].append(pred_degree_dist_sum / num_batches)
        train_history['target_degree_dist'].append(target_degree_dist)
        
        # Evaluate on validation set
        val_results = evaluate_model(
            model, validation_loader, max_nodes, node_feature_dim,
            current_beta, target_degree_dist, config, device
        )
        
        for k, v in val_results.items():
            if k in val_history:
                val_history[k].append(v)
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"--- Epoch {epoch+1}/{num_epochs} --- Beta: {current_beta:.4f}")
            print(f"  Train | Loss: {train_history['total'][-1]:.4f} | "
                  f"Recon: {train_history['reconstruction'][-1]:.4f} | "
                  f"KL: {train_history['kl'][-1]:.4f}")
            print(f"        | Degree Dist: {train_history['degree_dist'][-1]:.4f} | "
                  f"Valency: {train_history['valency'][-1]:.4f} | "
                  f"Sparsity: {train_history['sparsity'][-1]:.4f} | "
                  f"Conn: {train_history['connectivity'][-1]:.4f}")
            
            print(f"  Valid | Loss: {val_history['total'][-1]:.4f} | "
                  f"Recon: {val_history['reconstruction'][-1]:.4f} | "
                  f"KL: {val_history['kl'][-1]:.4f}")
            print(f"        | Degree Dist: {val_history['degree_dist'][-1]:.4f} | "
                  f"Valency: {val_history['valency'][-1]:.4f} | "
                  f"Sparsity: {val_history['sparsity'][-1]:.4f} | "
                  f"Conn: {val_history['connectivity'][-1]:.4f} | "
                  f"Disconn: {val_history['disconnected_ratio'][-1]:.2%}")
            
            # Update plots
            plot_curves(train_history, val_history, epoch)
    
    # === Save Model ===
    print(f"Training finished. Saving model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)
    plt.ioff()
    plt.close('all')
    return model

if __name__ == "__main__":
    # Calculate target edge density
    dataset, _, max_nodes, target_degree_dist, _ = load_data()
    total_edges = 0
    total_possible_edges = 0
    
    for data in dataset:
        n_edges = data.edge_index.shape[1] // 2  # Each edge counted twice
        n_nodes = data.num_nodes
        total_edges += n_edges
        total_possible_edges += (n_nodes * (n_nodes - 1)) // 2
    
    target_edge_density = total_edges / total_possible_edges
    print(f"Target edge density: {target_edge_density:.4f}")
    
    # Configuration
    config = {
        'hidden_dim': 64,
        'latent_dim': 32,
        'learning_rate': 0.0005,
        'num_epochs': 400,
        'batch_size': 32,
        'num_layers': 3,
        'model_save_path': 'graph_vae_model.pt',
        
        # KL annealing
        'kl_annealing_start': 0,
        'kl_annealing_end': 100,
        'final_beta': 0.1,
        
        # Structure penalty weights
        'distribution_weight': 200.0,
        'valency_weight': 50.0,
        'sparsity_weight': 100.0,
        'connectivity_weight': 100.0,
        
        # Structure constraints
        'max_degree': 3,
        'target_edge_density': target_edge_density,
        'target_degree_dist': target_degree_dist,
    }
    
    # Run training
    trained_model = train_graph_vae(config)
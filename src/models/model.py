import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from src.models.layers import GraphMessagePassing

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    def sample(self, num_samples=1, node_counts=None, enforce_connectivity=True, 
               target_degree_matching=True, use_spanning_tree=True):
        """
        Sample new graphs from the prior with optional post-processing steps
        
        Post-processing methods:
        - enforce_connectivity: Ensures the generated graph is connected
        - target_degree_matching: Matches nodes to target degree distribution
        - use_spanning_tree: Creates a spanning tree to ensure connectivity (subset of enforce_connectivity)
        
        Args:
            num_samples: Number of graphs to sample
            node_counts: Optional list of node counts for each graph. If not provided, uses max_nodes.
            enforce_connectivity: Whether to enforce graph connectivity
            target_degree_matching: Whether to match target degree distribution
            use_spanning_tree: Whether to use spanning tree algorithm for connectivity
        
        Returns:
            node_features: Node features for sampled graphs [B, N, F]
            sampled_adj: Sampled adjacency matrices [B, N, N]
            node_mask: Mask indicating which nodes are valid [B, N]
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
        node_mask = torch.zeros(batch_size, self.max_nodes, dtype=torch.float32, device=device)
        
        # Process node counts
        node_counts = self._prepare_node_counts(num_samples, node_counts)
        
        # Apply post-processing for each graph in the batch
        for b in range(batch_size):
            n_nodes = node_counts[b]
            node_mask[b, :n_nodes] = 1.0
            
            # Generate target degree distribution if needed
            target_degrees = None
            if target_degree_matching:
                target_degrees = self._generate_target_degrees(n_nodes)
                
            # Sample adjacency matrix with optional post-processing
            sampled_adj[b] = self._sample_adjacency(
                adj_probs[b], n_nodes, target_degrees,
                enforce_connectivity, use_spanning_tree
            )
        
        return node_features, sampled_adj, node_mask
    
    def _prepare_node_counts(self, num_samples, node_counts=None):
        """Prepare the node counts for each graph in the batch"""
        if node_counts is None:
            # If not provided, use max nodes for all graphs
            node_counts = [self.max_nodes] * num_samples
        
        # Make sure node_counts is the right size
        if len(node_counts) != num_samples:
            node_counts = node_counts[:num_samples]
            # If still not enough, pad with max_nodes
            if len(node_counts) < num_samples:
                node_counts.extend([self.max_nodes] * (num_samples - len(node_counts)))
        
        return node_counts
    
    def _generate_target_degrees(self, n_nodes):
        """Generate target degrees for a graph with n_nodes"""
        # Ideal degree distribution (from MUTAG dataset)
        # We want about 20% degree 1, 40% degree 2, 40% degree 3
        target_degree_percentages = [0.0, 0.20, 0.40, 0.40, 0.0, 0.0, 0.0]  # [deg 0, 1, 2, 3, 4, 5, 6+]
        
        # Create target degree distribution for this graph
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
        
        return torch.tensor(target_degrees, device=device)
    
    def _sample_adjacency(self, adj_probs, n_nodes, target_degrees=None, 
                          enforce_connectivity=True, use_spanning_tree=True):
        """
        Sample adjacency matrix with optional post-processing steps
        
        Args:
            adj_probs: Edge probabilities [N, N]
            n_nodes: Number of nodes in the graph
            target_degrees: Target degree for each node, or None if not using target degree matching
            enforce_connectivity: Whether to enforce graph connectivity
            use_spanning_tree: Whether to use spanning tree algorithm for connectivity
        
        Returns:
            sampled_adj: Sampled adjacency matrix [N, N]
        """
        # Initialize adjacency matrix
        sampled_adj = torch.zeros_like(adj_probs)
        
        # Create a list of potential edges sorted by probability (only for valid nodes)
        edge_list = []
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):  # Upper triangle only
                edge_list.append((i, j, adj_probs[i, j].item()))
        
        # Sort by probability (highest first)
        edge_list.sort(key=lambda x: x[2], reverse=True)
        
        # Track current node degrees
        degrees = torch.zeros(n_nodes, device=device)
        
        # If we're enforcing connectivity, we need to keep track of connected components
        if enforce_connectivity:
            if use_spanning_tree:
                sampled_adj = self._apply_spanning_tree(sampled_adj, edge_list, n_nodes, degrees, target_degrees)
            
            # Ensure the graph is fully connected
            sampled_adj = self._ensure_connectivity(sampled_adj, n_nodes, degrees, target_degrees)
        
        # Add additional edges to match target degrees if specified
        if target_degrees is not None:
            sampled_adj = self._match_target_degrees(sampled_adj, n_nodes, degrees, target_degrees, adj_probs)
        # If no post-processing, just sample based on probabilities
        elif not enforce_connectivity:
            sampled_adj = self._simple_probability_sampling(adj_probs, n_nodes)
            
        return sampled_adj
    
    def _apply_spanning_tree(self, sampled_adj, edge_list, n_nodes, degrees, target_degrees=None):
        """Create a spanning tree to ensure connectivity"""
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
            # Check if we can add this edge based on target degrees (if any)
            if target_degrees is None or (degrees[i] < target_degrees[i] and degrees[j] < target_degrees[j]):
                sampled_adj[i, j] = 1.0
                sampled_adj[j, i] = 1.0
                degrees[i] += 1
                degrees[j] += 1
                union(i, j)
                break
        
        # Continue building spanning tree (connect all components)
        edges_needed = n_nodes - 1  # Spanning tree needs n-1 edges
        edges_added = 1
        
        for i, j, prob in edge_list:
            # Skip if already added
            if sampled_adj[i, j] > 0:
                continue
                
            # Only add edge if it connects two separate components
            if find(i) != find(j):
                # Check if we can add this edge based on target degrees
                if target_degrees is None or (degrees[i] < target_degrees[i] and degrees[j] < target_degrees[j]):
                    sampled_adj[i, j] = 1.0
                    sampled_adj[j, i] = 1.0
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
                    
        return sampled_adj
    
    def _ensure_connectivity(self, sampled_adj, n_nodes, degrees, target_degrees=None):
        """Force graph connectivity, potentially overriding degree constraints"""
        # Use Union-Find data structure
        parents = list(range(n_nodes))
        
        def find(x):
            if parents[x] != x:
                parents[x] = find(parents[x])
            return parents[x]
        
        def union(x, y):
            parents[find(x)] = find(y)
        
        # Initialize parents based on existing edges
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if sampled_adj[i, j] > 0:
                    union(i, j)
        
        # Identify components
        component_to_nodes = {}
        for node in range(n_nodes):
            comp = find(node)
            if comp not in component_to_nodes:
                component_to_nodes[comp] = []
            component_to_nodes[comp].append(node)
        
        # If multiple components, connect them
        if len(component_to_nodes) > 1:
            components_list = list(component_to_nodes.keys())
            for i in range(1, len(components_list)):
                # Try to find nodes that can be connected
                found_connection = False
                
                # Try to respect degree constraints if provided
                if target_degrees is not None:
                    for node1 in component_to_nodes[components_list[0]]:
                        if degrees[node1] >= target_degrees[node1]:
                            continue
                        for node2 in component_to_nodes[components_list[i]]:
                            if degrees[node2] >= target_degrees[node2]:
                                continue
                                
                            # Connect these components
                            sampled_adj[node1, node2] = 1.0
                            sampled_adj[node2, node1] = 1.0
                            degrees[node1] += 1
                            degrees[node2] += 1
                            union(node1, node2)
                            found_connection = True
                            break
                        if found_connection:
                            break
                
                # If no connection possible with degree constraints, relax constraints
                if not found_connection:
                    node1 = component_to_nodes[components_list[0]][0]
                    node2 = component_to_nodes[components_list[i]][0]
                    sampled_adj[node1, node2] = 1.0
                    sampled_adj[node2, node1] = 1.0
                    degrees[node1] += 1
                    degrees[node2] += 1
                    union(node1, node2)
        
        return sampled_adj
    
    def _match_target_degrees(self, sampled_adj, n_nodes, degrees, target_degrees, adj_probs):
        """Add edges to match target degrees while respecting constraints"""
        for i in range(n_nodes):
            # Skip if already at or above target degree
            if degrees[i] >= target_degrees[i]:
                continue
            
            # Find potential neighbors for this node
            potential_neighbors = []
            for j in range(n_nodes):
                if i != j and sampled_adj[i, j] == 0 and degrees[j] < target_degrees[j]:
                    potential_neighbors.append((j, adj_probs[i, j].item()))
            
            # Sort by probability
            potential_neighbors.sort(key=lambda x: x[1], reverse=True)
            
            # Add edges until reaching target degree
            for j, prob in potential_neighbors:
                if degrees[i] >= target_degrees[i] or degrees[j] >= target_degrees[j]:
                    continue
                    
                # Add edge
                sampled_adj[i, j] = 1.0
                sampled_adj[j, i] = 1.0
                degrees[i] += 1
                degrees[j] += 1
                
                if degrees[i] >= target_degrees[i]:
                    break
                    
        return sampled_adj
    
    def _simple_probability_sampling(self, adj_probs, n_nodes):
        """Sample edges based purely on probabilities"""
        # Make a copy to avoid modifying the original
        sampled_adj = torch.zeros_like(adj_probs)
        
        # Simple threshold sampling for valid nodes
        threshold = 0.5  # Can be adjusted
        sampled_adj[:n_nodes, :n_nodes] = (adj_probs[:n_nodes, :n_nodes] > threshold).float()
        
        # Make adjacency matrix symmetric (undirected graph)
        sampled_adj = torch.triu(sampled_adj, diagonal=1) + torch.triu(sampled_adj, diagonal=1).transpose(0, 1)
        
        # No self-loops
        sampled_adj.fill_diagonal_(0)
        
        return sampled_adj

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
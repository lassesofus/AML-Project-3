import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import GraphConvLayer, GraphAttention

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Graph Encoder
class GraphEncoder(nn.Module):
    """Encoder for Graph VAE with graph-level latent representation"""
    
    def __init__(self, node_feature_dim, hidden_dim, latent_dim, num_layers=3):
        super().__init__()
        
        # Initial feature projection
        self.input_net = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Graph convolutional layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(GraphConvLayer(hidden_dim, hidden_dim))
        
        # Graph-level pooling output projections
        self.mean_net = nn.Linear(hidden_dim, latent_dim)
        self.logvar_net = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x, edge_index, batch):
        # Initial projection
        h = self.input_net(x)
        
        # Apply graph convolutions
        for layer in self.conv_layers:
            h = layer(h, edge_index)
            h = F.relu(h)
        
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
    
    def __init__(self, latent_dim, hidden_dim, max_nodes=28, node_feature_dim=7):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim
        
        # Latent to graph representation
        self.latent_to_graph = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Graph representation to initial node embeddings
        self.init_node_embeddings = nn.Linear(hidden_dim, max_nodes * hidden_dim)
        
        # Attention layers to model node dependencies
        self.attention1 = GraphAttention(hidden_dim, hidden_dim, heads=4)
        self.attention2 = GraphAttention(hidden_dim, hidden_dim, heads=4)
        
        # Additional layer for connectivity awareness
        self.connectivity_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Final projections
        self.node_features_generator = nn.Linear(hidden_dim, node_feature_dim)
        self.edge_weight_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z, node_counts=None):
        batch_size = z.size(0)
        
        # Transform latent to graph representation
        graph_repr = self.latent_to_graph(z)
        
        # Generate initial node embeddings
        node_embeddings = self.init_node_embeddings(graph_repr)
        node_embeddings = node_embeddings.view(batch_size, self.max_nodes, -1)
        
        # Apply attention layers for modeling node dependencies
        node_embeddings1, attn1 = self.attention1(node_embeddings)
        node_embeddings = node_embeddings + node_embeddings1  # Residual connection
        
        node_embeddings2, attn2 = self.attention2(node_embeddings)
        node_embeddings = node_embeddings + node_embeddings2  # Residual connection
        
        # Apply connectivity awareness layer
        node_embeddings = node_embeddings + self.connectivity_layer(node_embeddings)
        
        # Generate node features
        node_features = self.node_features_generator(node_embeddings)
        
        # Generate edge logits using pairwise node embedding similarity
        edge_logits = torch.zeros(batch_size, self.max_nodes, self.max_nodes, device=z.device)
        
        for b in range(batch_size):
            for i in range(self.max_nodes):
                for j in range(i+1, self.max_nodes):  # Only upper triangular (undirected graph)
                    # Combine node embeddings to predict edge
                    combined = torch.cat([node_embeddings[b, i], node_embeddings[b, j]], dim=0)
                    edge_weight = self.edge_weight_generator(combined.unsqueeze(0)).squeeze()
                    
                    # Symmetric matrix
                    edge_logits[b, i, j] = edge_weight
                    edge_logits[b, j, i] = edge_weight
        
        # Make adjacency matrix symmetric (undirected graph)
        # Set diagonal to very negative values (no self-loops)
        mask = torch.eye(self.max_nodes, device=edge_logits.device).unsqueeze(0)
        edge_logits = edge_logits * (1 - mask) - 1e9 * mask
        
        # Return attention weights for visualization/analysis
        return node_features, edge_logits, {"attn1": attn1, "attn2": attn2}

# Complete Graph VAE
class GraphVAE(nn.Module):
    """Variational Autoencoder for Graphs with graph-level latent variable"""
    
    def __init__(self, node_feature_dim, hidden_dim, latent_dim, max_nodes=28, num_layers=3, target_degree_dist=None):
        super().__init__()
        
        self.encoder = GraphEncoder(node_feature_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = GraphDecoder(latent_dim, hidden_dim, max_nodes, node_feature_dim)
        self.latent_dim = latent_dim
        self.max_nodes = max_nodes
    
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
        node_features, adj_logits, attn_weights = self.decode(z, node_counts)
        
        return node_features, adj_logits, z_mean, z_logvar, attn_weights
    
    def sample(self, num_samples=1, node_counts=None):
        """
        Sample new graphs from the prior without post-processing
        
        Args:
            num_samples: Number of graphs to sample
            node_counts: Optional list of node counts for each graph
        
        Returns:
            node_features: Node features for sampled graphs [B, N, F]
            sampled_adj: Sampled adjacency matrices [B, N, N]
            node_mask: Mask indicating which nodes are valid [B, N]
        """
        # Sample from the latent prior
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Decode to get node features and edge logits
        node_features, adj_logits, _ = self.decode(z)
        
        # Calculate adjacency probabilities
        adj_probs = torch.sigmoid(adj_logits)
        
        # Prepare node mask and process node counts
        node_mask = torch.zeros(num_samples, self.max_nodes, dtype=torch.float32, device=device)
        
        if node_counts is None:
            node_counts = [self.max_nodes // 2] * num_samples
        else:
            if len(node_counts) < num_samples:
                node_counts.extend([self.max_nodes // 2] * (num_samples - len(node_counts)))
            node_counts = node_counts[:num_samples]
        
        # Set node masks
        for b in range(num_samples):
            n_nodes = min(node_counts[b], self.max_nodes) 
            node_mask[b, :n_nodes] = 1.0
        
        # Sample adjacency matrices
        sampled_adj = torch.zeros_like(adj_probs)
        
        for b in range(num_samples):
            n_nodes = int(node_mask[b].sum().item())
            if n_nodes <= 1:
                continue
            
            # Extract probabilities for this graph
            graph_probs = adj_probs[b, :n_nodes, :n_nodes]
            
            # Calculate the number of edges we expect based on original data
            # For molecular graphs, expect ~2 edges per node on average
            target_edge_count = int(1.8 * n_nodes)  # Average degree between 1.5-2 is common for molecular graphs
            
            # Sort all possible edges by probability
            triu_indices = torch.triu_indices(n_nodes, n_nodes, offset=1, device=device)
            edge_probs = graph_probs[triu_indices[0], triu_indices[1]]
            
            # Use a dynamic threshold based on edge probabilities
            if target_edge_count < len(edge_probs):
                # Sort probabilities and find threshold at target position
                sorted_probs, _ = torch.sort(edge_probs, descending=True)
                threshold = sorted_probs[min(target_edge_count, len(sorted_probs)-1)]
                threshold = max(threshold, 0.1)  # Set minimum threshold
            else:
                threshold = 0.1  # Default threshold
            
            # Apply threshold to get binary adjacency matrix
            binary_adj = (graph_probs > threshold).float()
            
            # Make symmetric (undirected graph)
            binary_adj = torch.maximum(binary_adj, binary_adj.transpose(0, 1))
            
            # Update the sampled adjacency matrix
            sampled_adj[b, :n_nodes, :n_nodes] = binary_adj
        
        return node_features, sampled_adj, node_mask
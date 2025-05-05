import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import GraphConvLayer

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
        
        # Graph representation to node features
        self.node_features_generator = nn.Linear(hidden_dim, max_nodes * node_feature_dim)
        
        # Graph representation to adjacency matrix
        self.edge_generator = nn.Linear(hidden_dim, max_nodes * max_nodes)
    
    def forward(self, z, node_counts=None):
        batch_size = z.size(0)
        
        # Transform latent to graph representation
        graph_repr = self.latent_to_graph(z)
        
        # Generate node features from graph representation
        node_features = self.node_features_generator(graph_repr)
        node_features = node_features.view(batch_size, self.max_nodes, self.node_feature_dim)
        
        # Generate adjacency matrix from graph representation
        edge_logits = self.edge_generator(graph_repr)
        edge_logits = edge_logits.view(batch_size, self.max_nodes, self.max_nodes)
        
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
        node_features, adj_logits = self.decode(z, node_counts)
        
        return node_features, adj_logits, z_mean, z_logvar
    
    def sample(self, num_samples=1, node_counts=None):
        """
        Sample new graphs from the prior
        
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
        node_features, adj_logits = self.decode(z)
        
        # Convert logits to adjacency matrix with a threshold
        adj_probs = torch.sigmoid(adj_logits)
        sampled_adj = (adj_probs > 0.5).float()
        
        # Prepare node mask
        node_mask = torch.zeros(num_samples, self.max_nodes, dtype=torch.float32, device=device)
        
        # Process node counts
        if node_counts is None:
            node_counts = [self.max_nodes // 2] * num_samples  # Use half of max nodes by default
        else:
            if len(node_counts) < num_samples:
                node_counts.extend([self.max_nodes // 2] * (num_samples - len(node_counts)))
            node_counts = node_counts[:num_samples]
        
        # Set node masks
        for b in range(num_samples):
            n_nodes = min(node_counts[b], self.max_nodes)
            node_mask[b, :n_nodes] = 1.0
        
        return node_features, sampled_adj, node_mask
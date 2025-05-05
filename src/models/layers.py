import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    """Graph Convolutional Layer (GCN) implementation without external dependencies"""
    
    def __init__(self, in_dim, out_dim, add_self_loops=True, normalize=True, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index):
        # Get device and node count
        device = x.device
        num_nodes = x.size(0)
        
        # Add self-loops if requested
        if self.add_self_loops:
            self_loops = torch.arange(num_nodes, device=device)
            self_loops = self_loops.unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_loops], dim=1)
        
        # Linear transformation of node features
        transformed_features = torch.matmul(x, self.weight)
        
        # Create a normalized adjacency matrix for GCN
        if self.normalize:
            # Compute node degrees for normalization
            row, col = edge_index[0], edge_index[1]
            deg = torch.zeros(num_nodes, device=device)
            deg = deg.scatter_add_(0, row, torch.ones(row.size(0), device=device))
            
            # D^(-1/2)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            
            # Apply normalization D^(-1/2) A D^(-1/2)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            norm = torch.ones(edge_index.size(1), device=device)
        
        # Aggregate messages from neighbors (GCN propagation)
        output = torch.zeros_like(transformed_features)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            output[dst] += norm[i] * transformed_features[src]
        
        # Add bias if specified
        if self.bias is not None:
            output += self.bias
        
        return output

class ConnectivityAwareLayer(nn.Module):
    """Layer that helps the model understand graph connectivity patterns"""
    
    def __init__(self, hidden_dim, steps=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.steps = steps
        
        # Message passing networks
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention mechanism to focus on important connections
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Global connectivity features
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x, edge_index):
        """
        Forward pass of connectivity-aware layer
        
        Args:
            x: Node features [N, D]
            edge_index: Edge indices [2, E]
            
        Returns:
            Enhanced node features [N, D]
        """
        num_nodes = x.size(0)
        device = x.device
        node_features = x
        
        # Compute global graph representation
        global_repr = torch.mean(x, dim=0, keepdim=True)
        global_repr = self.global_pool(global_repr)
        
        # Use multi-step propagation to capture long-range dependencies
        for step in range(self.steps):
            # Prepare messages
            src, dst = edge_index[0], edge_index[1]
            messages = torch.zeros(edge_index.size(1), self.hidden_dim, device=device)
            
            # Compute message for each edge
            for i in range(edge_index.size(1)):
                s, d = src[i], dst[i]
                edge_feat = torch.cat([node_features[s], node_features[d]], dim=0)
                messages[i] = self.message_net(edge_feat.unsqueeze(0)).squeeze(0)
                
            # Compute attention weights
            attention_weights = torch.zeros(edge_index.size(1), device=device)
            for i in range(edge_index.size(1)):
                s, d = src[i], dst[i]
                edge_feat = torch.cat([node_features[s], node_features[d]], dim=0)
                attention_weights[i] = self.attention(edge_feat.unsqueeze(0)).squeeze()
            
            # Apply attention and aggregate messages
            weighted_messages = messages * attention_weights.unsqueeze(1)
            aggregated = torch.zeros(num_nodes, self.hidden_dim, device=device)
            for i in range(edge_index.size(1)):
                aggregated[dst[i]] += weighted_messages[i]
            
            # Update node features
            new_features = torch.zeros_like(node_features)
            for i in range(num_nodes):
                # Combine node features with aggregated messages and global context
                combined = torch.cat([node_features[i], aggregated[i]], dim=0)
                new_features[i] = self.update_net(combined.unsqueeze(0)).squeeze(0)
            
            # Add global context and residual connection
            node_features = new_features + node_features + global_repr
        
        return node_features

# Keep the original message passing layer for backward compatibility
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
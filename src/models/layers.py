import torch
import torch.nn as nn

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
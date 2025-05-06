import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from scipy import sparse

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_reconstruction_loss(adj_logits, node_features, adj_target, node_target, node_mask):
    """
    Calculate reconstruction loss for adjacency matrix and node features
    
    Args:
        adj_logits: Predicted adjacency logits [B, N, N]
        node_features: Predicted node features [B, N, F]
        adj_target: Target adjacency matrix [B, N, N]
        node_target: Target node features [B, N, F]
        node_mask: Binary mask for real nodes [B, N]
        
    Returns:
        adj_recon_loss: Adjacency reconstruction loss
        node_recon_loss: Node feature reconstruction loss
    """
    epsilon = 1e-8
    
    # Adjacency reconstruction loss (BCE)
    mask_2d = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
    adj_recon_loss = F.binary_cross_entropy_with_logits(
        adj_logits, adj_target, reduction='none'
    )
    adj_recon_loss = (adj_recon_loss * mask_2d).sum() / (mask_2d.sum() + epsilon)
    
    # Node feature reconstruction loss (MSE)
    node_mask_flat = node_mask.view(-1, 1)
    node_pred_flat = node_features.view(-1, node_features.size(-1))
    node_target_flat = node_target.view(-1, node_target.size(-1))
    
    node_recon_loss = F.mse_loss(
        node_pred_flat * node_mask_flat,
        node_target_flat * node_mask_flat,
        reduction='sum'
    ) / (node_mask_flat.sum() + epsilon)
    
    return adj_recon_loss, node_recon_loss

def calculate_kl_divergence(z_mean, z_logvar, batch_size):
    """
    Calculate KL divergence between latent distribution and prior
    
    Args:
        z_mean: Latent mean [B, Z]
        z_logvar: Latent log-variance [B, Z]
        batch_size: Batch size
        
    Returns:
        kl_loss: KL divergence loss
    """
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp()) / batch_size
    return kl_loss

def calculate_connectivity_loss(adj_probs, node_mask):
    """
    Calculate a differentiable loss that promotes graph connectivity
    Uses mincuts to promote connectivity.
    
    Args:
        adj_probs: Adjacency matrix probabilities [B, N, N]
        node_mask: Binary mask for real nodes [B, N]
        
    Returns:
        connectivity_loss: Loss that promotes connectivity (lower is better)
    """
    batch_size = adj_probs.size(0)
    total_loss = 0.0
    
    for b in range(batch_size):
        # Extract the valid subgraph based on node mask
        n_nodes = int(node_mask[b].sum().item())
        if n_nodes <= 1:  # Skip if only one node
            continue
            
        # Extract the submatrix of adjacency probabilities
        sub_adj = adj_probs[b, :n_nodes, :n_nodes]
        
        # Create node degree vector (sum of edge probabilities)
        degrees = sub_adj.sum(dim=1)
        
        # Create Laplacian matrix: L = D - A
        laplacian = torch.diag(degrees) - sub_adj
        
        # Instead of calculating eigenvalues, use the Rayleigh quotient to approximate
        # the second smallest eigenvalue (algebraic connectivity)
        # Initialize with a random vector orthogonal to the all-ones vector
        v = torch.randn(n_nodes, device=adj_probs.device)
        v = v - v.mean()  # Make orthogonal to all-ones vector
        v = v / (v.norm() + 1e-8)  # Normalize
        
        # Power iteration for 5 steps to approximate the eigenvector
        for _ in range(5):
            v = laplacian @ v
            v = v - v.mean()  # Keep orthogonal to all-ones
            v_norm = v.norm() + 1e-8
            v = v / v_norm
        
        # Rayleigh quotient: v^T * L * v (approximates algebraic connectivity)
        conn_value = torch.sum(v * (laplacian @ v))
        
        # Maximize algebraic connectivity (minimize its negative)
        total_loss += torch.exp(-conn_value)
        
        # Add additional loss term to encourage a minimum number of edges
        min_edge_density = 1.8 / n_nodes  # Target average degree of 1.8 (ensure tree-like structure)
        current_density = sub_adj.sum() / (n_nodes * n_nodes)
        density_penalty = F.relu(min_edge_density - current_density) * 5.0
        total_loss += density_penalty
    
    # Average across batch
    return total_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=adj_probs.device)

def calculate_degree_distribution_loss(adj_probs, node_mask, target_degree_dist):
    """
    Calculate loss to guide the model toward the target degree distribution
    
    Args:
        adj_probs: Adjacency matrix probabilities [B, N, N]
        node_mask: Binary mask for real nodes [B, N]
        target_degree_dist: Target degree distribution
        
    Returns:
        degree_loss: Loss based on degree distribution difference
    """
    if target_degree_dist is None:
        return torch.tensor(0.0, device=adj_probs.device)
    
    batch_size = adj_probs.size(0)
    total_loss = 0.0
    
    # Convert target distribution to tensor if it's a dictionary
    if isinstance(target_degree_dist, dict):
        max_degree = max(target_degree_dist.keys())
        target_dist = torch.zeros(max_degree + 1, device=adj_probs.device)
        for degree, prob in target_degree_dist.items():
            target_dist[degree] = prob
    else:
        # Assume it's already a tensor
        target_dist = target_degree_dist.to(adj_probs.device)
        max_degree = len(target_dist) - 1
    
    for b in range(batch_size):
        n_nodes = int(node_mask[b].sum().item())
        if n_nodes <= 1:
            continue
            
        # Calculate expected degree distribution
        sub_adj = adj_probs[b, :n_nodes, :n_nodes]
        expected_degrees = sub_adj.sum(dim=1)
        
        # Create histogram of expected degrees with soft binning
        degree_dist = torch.zeros(max_degree + 1, device=adj_probs.device)
        
        for node_idx in range(n_nodes):
            node_degree = expected_degrees[node_idx]
            # Cap degree at max_degree
            node_degree = min(node_degree, max_degree)
            
            # Hard assignment to nearest integer
            lower_idx = int(node_degree)
            upper_idx = min(lower_idx + 1, max_degree)
            
            # Soft binning based on distance to neighbors
            lower_weight = 1.0 - (node_degree - lower_idx)
            upper_weight = 1.0 - lower_weight
            
            degree_dist[lower_idx] += lower_weight
            if upper_idx > lower_idx:
                degree_dist[upper_idx] += upper_weight
        
        # Normalize to get probability distribution
        degree_dist = degree_dist / n_nodes
        
        # Calculate KL divergence between distributions
        # Adding small epsilon for numerical stability
        eps = 1e-10
        kl_div = torch.sum(target_dist * torch.log((target_dist + eps) / (degree_dist + eps)))
        total_loss += kl_div
    
    return total_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=adj_probs.device)

def calculate_vae_loss(adj_logits, node_features, adj_target, node_target, node_mask,
                      z_mean, z_logvar, beta, target_degree_dist=None, config=None):
    """
    Calculate enhanced VAE loss with connectivity and degree distribution components
    
    Args:
        adj_logits: Predicted adjacency logits [B, N, N]
        node_features: Predicted node features [B, N, F]
        adj_target: Target adjacency matrix [B, N, N]
        node_target: Target node features [B, N, F]
        node_mask: Binary mask for real nodes [B, N]
        z_mean: Latent mean [B, Z]
        z_logvar: Latent log-variance [B, Z]
        beta: KL divergence weight
        target_degree_dist: Target degree distribution
        config: Dictionary with configurations
        
    Returns:
        Dictionary of loss components
    """
    batch_size = node_features.shape[0]
    
    # Reconstruction loss
    adj_recon_loss, node_recon_loss = calculate_reconstruction_loss(
        adj_logits, node_features, adj_target, node_target, node_mask
    )
    
    # Total reconstruction loss
    recon_loss = adj_recon_loss + node_recon_loss
    
    # KL divergence
    kl_loss = calculate_kl_divergence(z_mean, z_logvar, batch_size)
    
    # Get adjacency probabilities for additional losses
    adj_probs = torch.sigmoid(adj_logits)
    
    # Connectivity loss
    connectivity_loss = calculate_connectivity_loss(adj_probs, node_mask)
    connectivity_weight = 0.5 if config is None else config.get('connectivity_weight', 0.5)
    
    # Degree distribution loss
    degree_loss = calculate_degree_distribution_loss(adj_probs, node_mask, target_degree_dist)
    degree_weight = 0.6 if config is None else config.get('degree_weight', 0.6)
    
    # Total loss with all components
    total_loss = (
        recon_loss + 
        beta * kl_loss + 
        connectivity_weight * connectivity_loss + 
        degree_weight * degree_loss
    )
    
    # Return comprehensive loss dictionary
    return {
        'total': total_loss,
        'reconstruction': recon_loss,
        'kl': kl_loss,
        'connectivity': connectivity_loss,
        'degree': degree_loss
    }

def kl_annealing_factor(epoch, config):
    """
    Calculate KL annealing factor based on current epoch
    
    Args:
        epoch: Current epoch number
        config: Dictionary with annealing parameters
    
    Returns:
        Beta value for KL loss weighting
    """
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
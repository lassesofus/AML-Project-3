import torch
import torch.nn.functional as F
from src.utils.graph_utils import calculate_degree_metrics, count_connected_components

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    # This penalizes the total number of edges to encourage sparser graphs
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

# KL annealing scheduler
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
import torch
import torch.nn.functional as F

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

def calculate_vae_loss(adj_logits, node_features, adj_target, node_target, node_mask,
                      z_mean, z_logvar, beta, target_degree_dist=None, config=None):
    """
    Calculate basic VAE loss (reconstruction + KL divergence)
    
    Args:
        adj_logits: Predicted adjacency logits [B, N, N]
        node_features: Predicted node features [B, N, F]
        adj_target: Target adjacency matrix [B, N, N]
        node_target: Target node features [B, N, F]
        node_mask: Binary mask for real nodes [B, N]
        z_mean: Latent mean [B, Z]
        z_logvar: Latent log-variance [B, Z]
        beta: KL divergence weight
        target_degree_dist: Target degree distribution (unused)
        config: Dictionary with configurations (unused)
        
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
    
    # Total loss (just basic VAE loss)
    total_loss = recon_loss + beta * kl_loss
    
    # Return simplified loss dictionary with only essential components
    return {
        'total': total_loss,
        'reconstruction': recon_loss,
        'kl': kl_loss
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
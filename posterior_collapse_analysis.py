import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import load_model_checkpoint
from architecture import get_vae
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

def analyze_latent_space(model, dataloader, device):
    """
    Perform comprehensive analysis of the latent space to detect posterior collapse
    """
    model.eval()
    z_means = []
    z_stds = []
    graph_sizes = []
    
    print("Encoding dataset into latent space...")
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            # Encode graphs
            q = model.encoder(batch.x, batch.edge_index, batch.batch)
            z_means.append(q.base_dist.loc)
            z_stds.append(q.base_dist.scale)
            
            # Store graph sizes for coloring
            batch_sizes = torch.bincount(batch.batch)
            graph_sizes.append(batch_sizes)
    
    # Concatenate all latent vectors and sizes
    all_means = torch.cat(z_means, dim=0).cpu().numpy()
    all_stds = torch.cat(z_stds, dim=0).cpu().numpy()
    all_sizes = torch.cat(graph_sizes, dim=0).cpu().numpy()
    
    # 1. Calculate KL divergence from prior analytically for each data point
    # For Normal distributions: KL(q||p) = 0.5 * (trace(Σp^-1 * Σq) + (μp - μq)^T Σp^-1 (μp - μq) - k + log|Σp|/|Σq|)
    # With standard normal prior, this simplifies to:
    # KL(q||N(0,I)) = 0.5 * (sum(σq^2) + sum(μq^2) - k - sum(log(σq^2)))
    latent_dim = all_means.shape[1]
    kl_per_datapoint = 0.5 * (np.sum(all_stds**2, axis=1) + np.sum(all_means**2, axis=1) - latent_dim - np.sum(np.log(all_stds**2), axis=1))
    
    # 2. Visualize means and stds distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_means.flatten(), bins=50, alpha=0.7, label='Means')
    plt.axvline(0, color='r', linestyle='--')
    plt.title('Distribution of Posterior Means')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(all_stds.flatten(), bins=50, alpha=0.7, label='Std Devs')
    plt.axvline(1, color='r', linestyle='--')
    plt.title('Distribution of Posterior Standard Deviations')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('epochs/posterior_distribution.png')
    plt.close()
    
    # 3. Visualize latent space using PCA and t-SNE
    # PCA first
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(all_means)
    
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=all_sizes, cmap='viridis', 
                         alpha=0.7, s=50, edgecolors='w')
    plt.colorbar(scatter, label='Graph Size (nodes)')
    plt.title('PCA Projection of Latent Space')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # t-SNE for better cluster visualization
    print("Computing t-SNE projection (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(all_means)
    
    plt.subplot(2, 1, 2)
    scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=all_sizes, cmap='viridis', 
                         alpha=0.7, s=50, edgecolors='w')
    plt.colorbar(scatter, label='Graph Size (nodes)')
    plt.title('t-SNE Projection of Latent Space')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('epochs/latent_space_projections.png')
    plt.close()
    
    # 4. Compute and visualize per-dimension activity
    dim_variance = np.var(all_means, axis=0)
    dim_kl_contribution = np.mean(all_means**2, axis=0) + np.mean(all_stds**2, axis=0) - 1 - np.mean(np.log(all_stds**2), axis=0)
    
    # Sort dimensions by variance
    sorted_indices = np.argsort(dim_variance)[::-1]
    sorted_variance = dim_variance[sorted_indices]
    sorted_kl = dim_kl_contribution[sorted_indices]
    
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(latent_dim), sorted_variance)
    plt.axhline(y=0.01, color='r', linestyle='--', label='Collapse Threshold')
    plt.title('Latent Dimension Variances')
    plt.xlabel('Latent Dimension (sorted)')
    plt.ylabel('Variance')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(range(latent_dim), sorted_kl)
    plt.title('KL Contribution per Dimension')
    plt.xlabel('Latent Dimension (sorted by variance)')
    plt.ylabel('KL Contribution')
    
    plt.tight_layout()
    plt.savefig('epochs/latent_dimensions_activity.png')
    plt.close()
    
    # 5. Generate a heatmap of the correlation between dimensions
    corr_matrix = np.corrcoef(all_means, rowvar=False)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False, square=True)
    plt.title('Latent Dimension Correlation Matrix')
    plt.tight_layout()
    plt.savefig('epochs/latent_correlation_matrix.png')
    plt.close()
    
    # 6. Summary statistics
    active_dims = np.sum(dim_variance > 0.01)
    percent_active = 100 * active_dims / latent_dim
    avg_kl = np.mean(kl_per_datapoint)
    
    print(f"\n===== Posterior Collapse Analysis =====")
    print(f"Latent dimensions: {latent_dim}")
    print(f"Active dimensions: {active_dims} ({percent_active:.1f}%)")
    print(f"Average KL divergence: {avg_kl:.4f}")
    
    if active_dims < latent_dim * 0.5:
        print("\n⚠️ WARNING: Possible posterior collapse detected!")
        print("Less than 50% of latent dimensions are active.")
    
    if avg_kl < 1.0:
        print("\n⚠️ WARNING: Low KL divergence suggests potential posterior collapse.")
    
    return {
        'active_dims': active_dims,
        'percent_active': percent_active,
        'avg_kl': avg_kl,
        'dim_variance': dim_variance,
        'kl_per_datapoint': kl_per_datapoint
    }

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs('epochs', exist_ok=True)
    
    # Load dataset
    device = 'cpu'
    dataset = TUDataset(root='./data/', name='MUTAG').to(device)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Load the trained model
    num_nodes_list = [data.num_nodes for data in dataset]
    vae = get_vae(num_nodes_list)
    model_path = './models/graph_vae.pt'
    
    try:
        vae = load_model_checkpoint(vae, model_path, device=device)
        print(f"Analyzing model from {model_path}")
        
        # Run analysis
        stats = analyze_latent_space(vae, dataloader, device)
        
        # Save analysis results
        np.save('epochs/posterior_collapse_stats.npy', stats)
        print(f"Analysis complete! See 'epochs' folder for visualization outputs.")
        
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first or provide the correct path.")

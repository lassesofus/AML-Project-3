from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from vae import VAE
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx, degree
from torch_geometric.utils import to_dense_adj
import networkx as nx
import pdb
import numpy as np

def calculate_degree_penalty(adj_matrices, num_nodes_per_graph=None, max_degree=3, progressive=True):
    """
    Computes a penalty based on how many nodes have degree > max_degree
    Args:
        adj_matrices: Batch of adjacency matrices [batch_size, num_nodes, num_nodes]
        num_nodes_per_graph: Optional tensor with number of nodes per graph in batch
        max_degree: Maximum allowed degree before penalty is applied
        progressive: If True, apply progressively higher penalties for higher degrees
    Returns:
        Penalty term to add to loss
    """
    # Infer per-graph node counts if not provided
    B, N, _ = adj_matrices.shape
    if num_nodes_per_graph is None:
        num_nodes_per_graph = torch.full((B,), N, dtype=torch.long, device=adj_matrices.device)

    # Binarize adjacency for integer edge counts
    bin_adj = (adj_matrices > 0.5).float()
    # Compute degree per node
    deg = bin_adj.sum(dim=2)        # [B, N]
    # Build mask to ignore padded nodes
    idx = torch.arange(N, device=adj_matrices.device).unsqueeze(0)  # [1, N]
    mask = (idx < num_nodes_per_graph.unsqueeze(1)).float()        # [B, N]
    
    # Compute excess degree over max_degree, only for real nodes
    if progressive:
        # Apply progressively higher penalties for higher degrees
        # Square the excess to make higher degrees contribute quadratically more
        raw_excess = torch.clamp(deg - max_degree, min=0)
        excess = torch.pow(raw_excess, 2) * mask
    else:
        # Original linear penalty
        excess = torch.clamp(deg - max_degree, min=0) * mask
        
    # Normalize per graph by its real node count
    per_graph = excess.sum(dim=1) / mask.sum(dim=1).clamp(min=1)    # [B]
    # Return mean penalty over batch
    return per_graph.mean()


def analyze_and_compare_graphs(raw_samples, cc_samples, epoch, calculate_degree_penalty, debug_mode=False):
    """
    Analyze and visualize both raw sampled graphs and their largest connected components
    to understand degree distribution differences.
    
    Args:
        raw_samples: List of adjacency matrices before connected component extraction
        cc_samples: List of adjacency matrices after connected component extraction
        epoch: Current training epoch  
        calculate_degree_penalty: Function to calculate degree penalty
        debug_mode: Whether to print detailed debug information
    """
    import matplotlib as mpl
    
    # Create figure with enough space for both raw and CC graphs
    n_samples = len(raw_samples)
    fig = plt.figure(figsize=(5*n_samples, 10))
    gs = plt.GridSpec(3, n_samples, height_ratios=[10, 10, 1])
    
    # Find global max degree across all graphs
    global_max_degree = 0
    for A in raw_samples + cc_samples:
        degrees = torch.sum(A, dim=1).cpu().numpy()
        global_max_degree = max(global_max_degree, int(degrees.max()))
    
    # Create a consistent colormap
    norm = mpl.colors.Normalize(vmin=0, vmax=max(1, global_max_degree))
    cmap = plt.get_cmap('viridis')
    
    # Plot raw graphs (top row)
    for i, A in enumerate(raw_samples):
        ax = fig.add_subplot(gs[0, i])
        
        degrees = torch.sum(A, dim=1).cpu().numpy()
        max_degree = degrees.max()
        avg_degree = degrees.mean()
        degree_counts = {d: (degrees == d).sum() for d in range(int(max_degree) + 1)}
        
        penalty = calculate_degree_penalty(A.unsqueeze(0)).item()
        
        if debug_mode:
            print(f"Raw Graph {i+1}:")
            print(f"  Degree penalty: {penalty:.4f}")
            print(f"  Max degree: {max_degree}")
            print(f"  Avg degree: {avg_degree:.2f}")
            print(f"  Degree distribution: {degree_counts}")
        
        graph_nx = nx.from_numpy_array(A.cpu().numpy())
        node_degrees = dict(graph_nx.degree())
        node_colors = [node_degrees[n] for n in graph_nx.nodes()]
        
        nx.draw(graph_nx, ax=ax, node_size=40, with_labels=False,
                node_color=node_colors, arrows=False, cmap='viridis',
                vmin=0, vmax=max(1, global_max_degree))
        
        ax.set_title(f"Raw Graph {i+1} - Max deg: {max_degree:.0f}", fontsize=10)
        ax.axis('off')
    
    # Plot connected component graphs (middle row)
    for i, A in enumerate(cc_samples):
        ax = fig.add_subplot(gs[1, i])
        
        degrees = torch.sum(A, dim=1).cpu().numpy()
        max_degree = degrees.max()
        avg_degree = degrees.mean()
        degree_counts = {d: (degrees == d).sum() for d in range(int(max_degree) + 1)}
        
        penalty = calculate_degree_penalty(A.unsqueeze(0)).item()
        
        if debug_mode:
            print(f"CC Graph {i+1}:")
            print(f"  Degree penalty: {penalty:.4f}")
            print(f"  Max degree: {max_degree}")
            print(f"  Avg degree: {avg_degree:.2f}")
            print(f"  Degree distribution: {degree_counts}")
        
        graph_nx = nx.from_numpy_array(A.cpu().numpy())
        node_degrees = dict(graph_nx.degree())
        node_colors = [node_degrees[n] for n in graph_nx.nodes()]
        
        nx.draw(graph_nx, ax=ax, node_size=40, with_labels=False,
                node_color=node_colors, arrows=False, cmap='viridis',
                vmin=0, vmax=max(1, global_max_degree))
        
        ax.set_title(f"Connected Component {i+1} - Max deg: {max_degree:.0f}", fontsize=10)
        ax.axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_subplot(gs[2, :])
    cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                                 orientation='horizontal', label='Node Degree')
    cb.set_ticks(range(global_max_degree + 1))
    cb.set_ticklabels(range(global_max_degree + 1))
    
    plt.tight_layout()
    plt.savefig(f'epochs/graph_comparison_epoch_{epoch}.png')
    plt.close()

def visualize_latent_space(model, dataloader, epoch, device):
    """
    Visualize the latent space by encoding graphs and plotting their 2D PCA projection
    """
    model.eval()
    z_list = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            # Encode graphs
            q = model.encoder(batch.x, batch.edge_index, batch.batch)
            z = q.mean  # Use mean of the posterior
            z_list.append(z)
            # For visualization purposes, use graph size as label
            graph_sizes = torch.bincount(batch.batch)
            labels.append(graph_sizes)
    
    # Concatenate all latent vectors and labels
    z_all = torch.cat(z_list, dim=0).cpu().numpy()
    labels_all = torch.cat(labels, dim=0).cpu().numpy()
    
    # Apply PCA to reduce to 2 dimensions for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_all)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=labels_all, cmap='viridis', 
                         alpha=0.6, s=50, edgecolors='w')
    plt.colorbar(scatter, label='Graph Size (nodes)')
    plt.title(f'Latent Space Visualization (Epoch {epoch})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'epochs/latent_space_epoch_{epoch}.png')
    plt.close()

def calculate_active_units(model, dataloader, device, threshold=0.01):
    """
    Calculate the number of active units in the latent space
    An active unit has variance across the batch greater than the threshold
    """
    model.eval()
    z_means = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            # Encode graphs
            q = model.encoder(batch.x, batch.edge_index, batch.batch)
            z_mean = q.mean
            z_means.append(z_mean)
    
    # Concatenate all means
    all_means = torch.cat(z_means, dim=0)
    
    # Calculate variance for each dimension
    latent_variances = all_means.var(dim=0)
    
    # Count dimensions with variance above threshold
    active_dims = (latent_variances > threshold).sum().item()
    
    return active_dims, latent_variances.cpu().numpy()

def train_vae(model: VAE, dataloader, epochs=50, lr=1e-3, save_path='graph_vae.pt', 
              loss_plot_path='loss_curve.png', device='cpu', degree_penalty_weight=0.5, 
              debug_mode=False, patience=20, min_delta=0.0001,
              kl_annealing=True, min_kl_weight=0.0, max_kl_weight=1.0,
              free_bits=0.01, beta_vae=0.2):
    """
    Train a VAE model on graph data.
    
    Args:
        model: The VAE model to train
        dataloader: DataLoader for the training data
        epochs: Number of training epochs
        lr: Learning rate
        save_path: Path to save the trained model
        loss_plot_path: Path to save the loss curve plot
        device: Device to run the training on
        degree_penalty_weight: Weight for the degree penalty term
        debug_mode: Whether to print debugging information and create sample visualizations
        patience: Number of epochs to wait for improvement before early stopping
        min_delta: Minimum change in loss to qualify as improvement
        kl_annealing: Whether to use KL annealing during training
        min_kl_weight: Minimum KL weight for annealing
        max_kl_weight: Maximum KL weight for annealing
        free_bits: Minimum KL per dimension to prevent posterior collapse
        beta_vae: Beta value for beta-VAE (values < 1 emphasize reconstruction)
    """
    model.to(device)
    
    # Use a learning rate scheduler for better convergence
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, 
        min_lr=1e-5, verbose=True
    )

    best_loss = float('inf')
    loss_history = []
    no_improvement_count = 0
    
    # Track degree penalties and connection sparsity separately
    degree_penalties = []
    kl_values = []
    
    # Track for posterior collapse diagnosis
    avg_kl_per_epoch = []
    active_units_per_epoch = []
    latent_dim = model.prior().base_dist.loc.shape[0]  # Get latent dimension size
    
    # Create a progress bar without postfix (we'll handle it separately)
    with tqdm(range(1, epochs + 1), desc="Training Epochs", ncols=100) as pbar:
        for epoch in pbar:
            model.train()
            total_loss = 0
            epoch_deg_penalty = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            
            # Cyclical KL annealing - works better than linear annealing
            if kl_annealing:
                # Use cosine annealing for smoother transitions
                cycle_progress = (epoch % 50) / 50  # Cycle every 50 epochs
                kl_weight = min_kl_weight + (max_kl_weight - min_kl_weight) * (1 - np.cos(np.pi * cycle_progress)) / 2
                kl_weight = beta_vae * kl_weight  # Apply beta-VAE scaling
            else:
                kl_weight = beta_vae

            for batch in dataloader:
                batch = batch.to(device)
                # Compute num_nodes_per_graph: count nodes per graph
                num_nodes_per_graph = torch.bincount(batch.batch)
                optimizer.zero_grad()
                A = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=model.node_dist.max_nodes)

                # Forward pass: compute ELBO losses
                loss, recon_loss, kl = model(batch.x, batch.edge_index, batch.batch, A, num_nodes_per_graph)
                
                # Apply free bits to prevent posterior collapse
                # This ensures each latent dimension contributes at least 'free_bits' to the KL term
                kl_per_dim = kl * latent_dim  # Scale up to get per-dimension KL
                kl_with_fb = torch.max(kl_per_dim, torch.tensor(free_bits * latent_dim).to(device)) 
                
                # Apply KL annealing weight with free bits
                loss = recon_loss + kl_weight * kl_with_fb
                
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl.item()

                # Re-encode + decode to get predicted adjacency
                q = model.encoder(batch.x, batch.edge_index, batch.batch)
                z = q.rsample()
                adj_pred_dist = model.decoder(z).base_dist
                
                # Get probability matrix
                adj_probs = torch.sigmoid(adj_pred_dist.logits)
                
                # For monitoring: Generate hard binary samples to calculate true degree penalty
                with torch.no_grad():
                    # Use Bernoulli sampling to get true binary edges (not probabilities)
                    rand = torch.rand_like(adj_probs)
                    adj_binary = (rand < adj_probs).float()
                    
                    # Print debug info occasionally if enabled
                    if debug_mode and epoch % 10 == 0 and torch.rand(1).item() < 0.05:
                        print(f"\nDEBUG - Epoch {epoch}")
                        print(f"Binary adj max value: {adj_binary.max().item()}")
                        print(f"Binary adj has {(adj_binary > 0).sum().item()} edges")
                        
                        # Calculate degree statistics
                        degrees = adj_binary.sum(dim=2)  # Sum along rows to get degrees
                        max_degree_val = degrees.max().item()
                        print(f"Max degree in batch: {max_degree_val}")
                        print(f"Degrees > 3: {(degrees > 3).sum().item()}")
                    
                    # Calculate degree penalty on actual binary matrices
                    true_deg_penalty = calculate_degree_penalty(adj_binary, num_nodes_per_graph=num_nodes_per_graph)
                    epoch_deg_penalty += true_deg_penalty.item()
                
                # Calculate penalty on probabilities for gradient computation
                # Use a lower temperature for sharper gradients at later epochs
                temp = max(0.1, 0.5 * (1 - epoch/epochs))
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(adj_probs)))
                soft_samples = torch.sigmoid((torch.log(adj_probs + 1e-10) - torch.log(1 - adj_probs + 1e-10) + gumbel_noise) / temp)
                
                # Apply penalty using differentiable samples
                deg_penalty_diff = calculate_degree_penalty(soft_samples, num_nodes_per_graph=num_nodes_per_graph)
                
                # Combine losses - don't increase degree penalty too much, it may interfere
                adaptive_penalty_weight = degree_penalty_weight * min(2.0, (1 + 0.2 * epoch / epochs))
                loss_with_penalty = loss + adaptive_penalty_weight * deg_penalty_diff
                
                # Add some L2 regularization to the decoder to prevent overfitting
                decoder_params = list(model.decoder.decoder_net.parameters())
                l2_reg = 1e-5 * sum(p.pow(2.0).sum() for p in decoder_params)
                loss_with_penalty = loss_with_penalty + l2_reg
                
                loss_with_penalty.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss_with_penalty.item() * batch.num_graphs

            avg_loss = total_loss / len(dataloader.dataset)
            avg_deg_penalty = epoch_deg_penalty / len(dataloader)
            avg_recon_loss = epoch_recon_loss / len(dataloader)
            avg_kl_loss = epoch_kl_loss / len(dataloader)
            loss_history.append(avg_loss)
            degree_penalties.append(avg_deg_penalty)
            kl_values.append(avg_kl_loss)
            
            # Add KL weight to tracked values for plotting
            avg_kl_per_epoch.append((avg_kl_loss, kl_weight))
            
            # Update learning rate based on validation loss
            scheduler.step(avg_loss)

            # Check for improvement
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                no_improvement_count = 0
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': best_loss,
                }, save_path)
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    tqdm.write(f"\nEarly stopping triggered after {epoch} epochs! No improvement for {patience} epochs.")
                    break

            # Update progress bar - keep it simple
            pbar.set_postfix({'Epoch': epoch, 'Loss': f'{avg_loss:.4f}'})
            
            # Print detailed stats on a second line
            tqdm.write(
                f"Epoch {epoch:3d} | "
                f"Loss: {avg_loss:.4f} | "
                f"Best: {best_loss:.4f} | "
                f"Recon: {avg_recon_loss:.4f} | "
                f"KL: {avg_kl_loss:.4f} (w={kl_weight:.2f}) | "
                f"Deg: {avg_deg_penalty:.4f} | "
                f"Pat: {no_improvement_count}/{patience} | "
                f"LR: {optimizer.param_groups[0]['lr']:.1e} | "
                f"FB: {free_bits:.4f}"
            )

            # Sample and visualize graphs at regular intervals or at the end
            if  (epoch % 20 == 0 or epoch == epochs):
                model.eval()
                with torch.no_grad():
                    # Sample graphs using the same method as in model.sample
                    n_samples = 6
                    train_samples = []
                    raw_samples = []
                    
                    for _ in range(n_samples):
                        # Sample from prior (same as in model.sample)
                        n = model.node_dist.sample(1)[0]
                        z = model.prior().sample(torch.Size([1]))
                        
                        # Get adjacency distribution and sample
                        adj_dist = model.decoder(z).base_dist
                        adj_probs = torch.sigmoid(adj_dist.logits)
                        raw_adj = adj_dist.sample()
                        raw_adj = raw_adj[0, :n, :n]
                        raw_adj.fill_diagonal_(0)
                        
                        # Store raw sample before connected component extraction
                        raw_samples.append(raw_adj)
                        
                        # Get largest connected component (as in model.sample)
                        A_np = raw_adj.detach().cpu().numpy()
                        G = nx.from_numpy_array(A_np)
                        largest_cc = max(nx.connected_components(G), key=len)
                        largest_cc = sorted(largest_cc)
                        lcc_adj = raw_adj[np.ix_(largest_cc, largest_cc)]
                        
                        train_samples.append(lcc_adj)
                    
                    # Calculate penalties individually for each graph
                    raw_penalties = [calculate_degree_penalty(s.unsqueeze(0)).item() for s in raw_samples]
                    cc_penalties = [calculate_degree_penalty(s.unsqueeze(0)).item() for s in train_samples]
                    
                    # Average the penalties
                    raw_penalty = sum(raw_penalties) / len(raw_penalties)
                    cc_penalty = sum(cc_penalties) / len(cc_penalties)
                    
                    if debug_mode:
                        print(f"\n--- Epoch {epoch} Sampling Comparison ---")
                        print(f"Training batch degree penalty: {avg_deg_penalty:.4f}")
                        print(f"Raw sampled graphs degree penalty: {raw_penalty:.4f}")
                        print(f"After connected component extraction: {cc_penalty:.4f}\n")
                        
                        # Show individual penalties for each graph
                        for i in range(len(raw_samples)):
                            print(f"Graph {i+1} - Raw penalty: {raw_penalties[i]:.4f}, CC penalty: {cc_penalties[i]:.4f}")
                    
                    # Visualize both versions to understand the difference
                    analyze_and_compare_graphs(raw_samples, train_samples, epoch, calculate_degree_penalty, debug_mode)
                
                # Visualize the latent space
                if debug_mode:
                    print(f"\n--- Epoch {epoch} Latent Space Visualization ---")
                    visualize_latent_space(model, dataloader, epoch, device)
                
                # Calculate active units
                active_units, variances = calculate_active_units(model, dataloader, device)
                active_units_per_epoch.append((epoch, active_units))
                
                # Log posterior collapse indicators
                active_percent = 100 * active_units / latent_dim
                print(f"\n--- Epoch {epoch} Posterior Collapse Check ---")
                print(f"KL Divergence: {avg_kl_loss:.4f} (weight={kl_weight:.2f})")
                print(f"Active dimensions: {active_units}/{latent_dim} ({active_percent:.1f}%)")
                
                # # Show variance distribution
                # plt.figure(figsize=(10, 5))
                # plt.bar(range(len(variances)), sorted(variances, reverse=True))
                # plt.axhline(y=0.01, color='r', linestyle='--', label='Threshold')
                # plt.title(f'Latent Dimension Variances (Epoch {epoch})')
                # plt.xlabel('Latent Dimension (sorted)')
                # plt.ylabel('Variance')
                # plt.legend()
                # plt.tight_layout()
                # plt.savefig(f'epochs/latent_variances_epoch_{epoch}.png')
                # plt.close()
                
    # Plot extended loss curve with KL and degree penalty
    plt.figure(figsize=(12, 8))
    
    # Plot main loss
    plt.subplot(3, 1, 1)
    plt.plot(range(1, len(loss_history) + 1), loss_history, label='Total Loss')
    plt.grid(True)
    plt.legend()
    plt.title('VAE Training Curves')
    
    # Plot KL divergence
    plt.subplot(3, 1, 2)
    plt.plot(range(1, len(kl_values) + 1), kl_values, label='KL Divergence', color='orange')
    plt.grid(True)
    plt.legend()
    
    # Plot degree penalty
    plt.subplot(3, 1, 3)
    plt.plot(range(1, len(degree_penalties) + 1), degree_penalties, label='Degree Penalty', color='green')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"📉 Saved extended loss curve to {loss_plot_path}")
    
    # Plot KL divergence separately with clearer indication of collapse
    plt.figure(figsize=(12, 6))
    epochs_list = range(1, len(avg_kl_per_epoch) + 1)
    kl_values = [kl for kl, _ in avg_kl_per_epoch]
    kl_weights = [w for _, w in avg_kl_per_epoch]
    
    # Plot KL values
    ax1 = plt.gca()
    ax1.plot(epochs_list, kl_values, 'b-', label='KL Divergence')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('KL Divergence', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.axhline(y=0.1, color='r', linestyle='--', label='Collapse Threshold')
    ax1.legend(loc='upper left')
    
    # Plot KL weights on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(epochs_list, kl_weights, 'g-', label='KL Weight')
    ax2.set_ylabel('KL Weight', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.legend(loc='upper right')
    
    # Add active units as annotations
    for epoch, active in active_units_per_epoch:
        idx = epoch - 1  # Convert to 0-based index
        if idx < len(epochs_list):
            ax1.annotate(f'{active}', 
                        (epochs_list[idx], kl_values[idx]),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')
    
    plt.title('KL Divergence and Active Units Over Training')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('epochs/posterior_collapse_monitoring.png')
    plt.close()
    
    print(f"📉 Saved posterior collapse diagnostics to epochs/posterior_collapse_monitoring.png")
    
    # Sample 100 graphs and analyze degree distribution
    print("\n📊 Sampling 100 graphs to analyze degree distribution...")
    model.eval()
    all_degrees = []
    total_nodes = 0
    
    with torch.no_grad():
        for i in range(100):
            sampled_graph = model.sample(1)[0]  # Get one graph
            degrees = torch.sum(sampled_graph, dim=1).int().cpu().numpy()
            all_degrees.extend(degrees)
            total_nodes += len(degrees)
    
    # Count occurrences of each degree
    unique_degrees, counts = np.unique(all_degrees, return_counts=True)
    degree_counts = dict(zip(unique_degrees, counts))
    
    print(f"\n===== Node Degree Distribution across 100 Sampled Graphs =====")
    print(f"Total nodes analyzed: {total_nodes}")
    
    # Print sorted by degree
    print("\nDegree Counts:")
    for degree in sorted(degree_counts.keys()):
        count = degree_counts[degree]
        percentage = 100 * count / total_nodes
        print(f"  Degree {degree}: {count} nodes ({percentage:.2f}%)")
    
    # Print summary statistics
    degrees_array = np.array(all_degrees)
    print("\nSummary Statistics:")
    print(f"  Mean degree: {degrees_array.mean():.2f}")
    print(f"  Median degree: {np.median(degrees_array):.2f}")
    print(f"  Min degree: {degrees_array.min()}")
    print(f"  Max degree: {degrees_array.max()}")
    print(f"  Standard deviation: {degrees_array.std():.2f}")
    
    # Generate a histogram of node degrees
    plt.figure(figsize=(10, 6))
    plt.hist(all_degrees, bins=range(int(max(all_degrees))+2), alpha=0.7, 
             edgecolor='black', linewidth=1.2)
    plt.title('Node Degree Distribution in Sampled Graphs')
    plt.xlabel('Node Degree')
    plt.ylabel('Count')
    plt.xticks(range(0, int(max(all_degrees))+1))
    plt.grid(alpha=0.3)
    plt.savefig('epochs/degree_distribution_histogram.png')
    plt.close()
    
    print(f"📈 Saved degree distribution histogram to epochs/degree_distribution_histogram.png")
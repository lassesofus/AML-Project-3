from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from vae import VAE
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx, degree
from torch_geometric.utils import to_dense_adj
import networkx as nx
import pdb

def calculate_degree_penalty(adj_matrices, num_nodes_per_graph=None, max_degree=3):
    """
    Computes a penalty based on how many nodes have degree > max_degree
    Args:
        adj_matrices: Batch of adjacency matrices [batch_size, num_nodes, num_nodes]
        num_nodes_per_graph: Optional tensor with number of nodes per graph in batch
        max_degree: Maximum allowed degree before penalty is applied
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
    excess = torch.clamp(deg - max_degree, min=0) * mask           # [B, N]
    # Normalize per graph by its real node count
    per_graph = excess.sum(dim=1) / mask.sum(dim=1).clamp(min=1)    # [B]
    # Return mean penalty over batch
    return per_graph.mean()

def analyze_and_visualize_graphs(sampled_graphs, epoch, calculate_degree_penalty):
    """
    Analyze node degrees and visualize sampled graphs
    
    Args:
        sampled_graphs: List of adjacency matrices representing the sampled graphs
        epoch: Current training epoch
        calculate_degree_penalty: Function to calculate degree penalty
    """
    # Debug: Calculate and print degree information for sampled graphs
    print(f"\n--- Epoch {epoch} Degree Analysis ---")
    
    # Find the global max degree for consistent color mapping across all graphs
    global_max_degree = 0
    for A in sampled_graphs:
        degrees = torch.sum(A, dim=1).cpu().numpy()
        global_max_degree = max(global_max_degree, int(degrees.max()))
    
    # Create a consistent colormap for all graphs
    import matplotlib as mpl        
    norm = mpl.colors.Normalize(vmin=0, vmax=max(1, global_max_degree))
    cmap = plt.get_cmap('viridis')
    
    # Analyze each graph
    for i, A in enumerate(sampled_graphs):
        # Calculate node degrees for this graph
        degrees = torch.sum(A, dim=1).cpu().numpy()
        max_degree = degrees.max()
        avg_degree = degrees.mean()
        degree_counts = {d: (degrees == d).sum() for d in range(int(max_degree) + 1)}
        
        # Calculate degree penalty for this graph
        sample_penalty = calculate_degree_penalty(A.unsqueeze(0)).item()
        
        print(f"Graph {i+1}:")
        print(f"  Degree penalty: {sample_penalty:.4f}")
        print(f"  Max degree: {max_degree}")
        print(f"  Avg degree: {avg_degree:.2f}")
        print(f"  Degree distribution: {degree_counts}")
    
    # Create a figure with subplots for graphs and a colorbar
    fig = plt.figure(figsize=(5*len(sampled_graphs), 6))
    
    # Create a grid specification for the layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, len(sampled_graphs), height_ratios=[10, 1])
    
    # Create axes for graphs
    axes = []
    for i in range(len(sampled_graphs)):
        axes.append(fig.add_subplot(gs[0, i]))
    
    # Plot each graph
    for i, ax in enumerate(axes):
        if i < len(sampled_graphs):
            A = sampled_graphs[i]
            graph_nx = nx.from_numpy_array(A.cpu().numpy())
            
            # Add degree information to the graph visualization
            node_degrees = dict(graph_nx.degree())
            node_colors = [node_degrees[n] for n in graph_nx.nodes()]
            
            nx.draw(graph_nx, ax=ax, node_size=40, with_labels=False, 
                    node_color=node_colors, arrows=False, cmap='viridis', 
                    vmin=0, vmax=max(1, global_max_degree))     
            
            ax.set_title(f"Graph {i+1} - Max deg: {max(node_degrees.values()) if node_degrees else 0}", fontsize=10)
            ax.axis('off')
    
    # Add a colorbar at the bottom spanning all graph plots
    cbar_ax = fig.add_subplot(gs[1, :])
    cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, 
                                  orientation='horizontal', label='Node Degree')
    
    # Add integer ticks to the colorbar
    cb.set_ticks(range(global_max_degree + 1))
    cb.set_ticklabels(range(global_max_degree + 1))

    plt.tight_layout()
    plt.savefig(f'epochs/sampled_graphs_epoch_{epoch}.png')
    plt.close()

def train_vae(model: VAE, dataloader, epochs=50, lr=1e-3, save_path='graph_vae.pt', loss_plot_path='loss_curve.png', device='cpu', degree_penalty_weight=0.1):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    loss_history = []
    with tqdm(range(1, epochs + 1), desc="Training Epochs") as pbar:
        for epoch in pbar:
            model.train()
            total_loss = 0

            for batch in dataloader:
                batch = batch.to(device)
                # Compute num_nodes_per_graph: count nodes per graph
                num_nodes_per_graph = torch.bincount(batch.batch)
                optimizer.zero_grad()
                A = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=model.node_dist.max_nodes)

                # Forward pass: compute ELBO losses
                loss, recon_loss, kl = model(batch.x, batch.edge_index, batch.batch, A, num_nodes_per_graph)

                # Re-encode + decode to get the model’s predicted adjacency
                q = model.encoder(batch.x, batch.edge_index, batch.batch)
                z = q.rsample()
                adj_pred_dist = model.decoder(z).base_dist       # Bernoulli distribution over edges
                adj_pred = torch.sigmoid(adj_pred_dist.logits)    # [batch_size, N, N] predicted adjacency probabilities

                # Compute degree penalty on the predicted adjacency
                deg_penalty = calculate_degree_penalty(adj_pred, num_nodes_per_graph=num_nodes_per_graph)
            
                # Combine losses
                loss_with_penalty = loss + degree_penalty_weight * deg_penalty
                
                loss_with_penalty.backward()
                optimizer.step()
                total_loss += loss_with_penalty.item() * batch.num_graphs

            avg_loss = total_loss / len(dataloader.dataset)
            loss_history.append(avg_loss)

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': best_loss,
                }, save_path)

            # Update progress bar with degree penalty information
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}', 
                'Best Loss': f'{best_loss:.4f}',
                'rec_loss': f'{recon_loss.item():.4f}',
                'kl_loss': f'{kl.item():.4f}',
                'deg_penalty': f'{deg_penalty.item():.4f}'
            })

            # Every tenth epoch, sample three graphs and visualize
            if epoch == 80:
                model.eval()
                with torch.no_grad():
                    sampled_graphs = model.sample(3)
                
                analyze_and_visualize_graphs(sampled_graphs, epoch, calculate_degree_penalty)
                
    # Plot loss curve
    plt.figure()
    plt.plot(range(1, epochs + 1), loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GraphVAE Training Loss Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"📉 Saved loss curve to {loss_plot_path}")
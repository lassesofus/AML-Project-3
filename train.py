import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj
import networkx as nx
from utils_viz import analyze_and_compare_graphs, plot_training_loss
from utils_stats import calculate_degree_penalty, compute_graph_statistics, calculate_active_units
from utils_stats import calculate_isolated_nodes_penalty, calculate_triangle_penalty

def train_vae(model, dataloader, epochs=50, lr=1e-3, save_path='graph_vae.pt', 
              loss_plot_path='loss_curve.png', device='cpu', degree_penalty_weight=0.5, 
              debug_mode=False, patience=20, min_delta=0.0001,
              kl_annealing=True, min_kl_weight=0.0, max_kl_weight=1.0,
              free_bits=0.01, beta_vae=0.2,
              isolated_nodes_penalty_weight=0.0, triangle_penalty_weight=0.0):
    """
    Train a VAE model on graph data with various regularization techniques.
    
    Additional Parameters:
        isolated_nodes_penalty_weight: Weight for penalizing isolated nodes (0.0 = off)
        triangle_penalty_weight: Weight for penalizing triangles (0.0 = off)
    """
    model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, 
        min_lr=1e-5, verbose=True
    )

    best_loss = float('inf')
    loss_history = []
    degree_penalties = []
    kl_values = []
    avg_kl_per_epoch = []
    active_units_per_epoch = []
    isolated_node_penalties = []
    triangle_penalties = []
    no_improvement_count = 0
    
    latent_dim = model.prior().base_dist.loc.shape[0]
    
    # Main training loop
    with tqdm(range(1, epochs + 1), desc="Training Epochs", ncols=100) as pbar:
        for epoch in pbar:
            model.train()
            total_loss = 0
            epoch_deg_penalty = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            epoch_isolated_penalty = 0
            epoch_triangle_penalty = 0
            
            # KL annealing schedule
            if kl_annealing:
                cycle_progress = (epoch % 50) / 50
                kl_weight = min_kl_weight + (max_kl_weight - min_kl_weight) * (1 - np.cos(np.pi * cycle_progress)) / 2
                kl_weight = beta_vae * kl_weight
            else:
                kl_weight = beta_vae

            for batch in dataloader:
                batch = batch.to(device)
                num_nodes_per_graph = torch.bincount(batch.batch)
                optimizer.zero_grad()
                A = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=model.node_dist.max_nodes)

                # Forward pass
                loss, recon_loss, kl = model(batch.x, batch.edge_index, batch.batch, A, num_nodes_per_graph)
                
                # Apply free bits to prevent posterior collapse
                kl_per_dim = kl * latent_dim
                kl_with_fb = torch.max(kl_per_dim, torch.tensor(free_bits * latent_dim).to(device)) 
                loss = recon_loss + kl_weight * kl_with_fb
                
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl.item()
                
                # Get decoded adjacency for penalties
                q = model.encoder(batch.x, batch.edge_index, batch.batch)
                z = q.rsample()
                adj_pred_dist = model.decoder(z).base_dist
                adj_probs = torch.sigmoid(adj_pred_dist.logits)
                
                # Calculate degree penalty
                with torch.no_grad():
                    rand = torch.rand_like(adj_probs)
                    adj_binary = (rand < adj_probs).float()
                    true_deg_penalty = calculate_degree_penalty(adj_binary, num_nodes_per_graph=num_nodes_per_graph)
                    epoch_deg_penalty += true_deg_penalty.item()
                
                # Differentiable degree penalty
                temp = max(0.1, 0.5 * (1 - epoch/epochs))
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(adj_probs)))
                soft_samples = torch.sigmoid((torch.log(adj_probs + 1e-10) - torch.log(1 - adj_probs + 1e-10) + gumbel_noise) / temp)
                deg_penalty_diff = calculate_degree_penalty(soft_samples, num_nodes_per_graph=num_nodes_per_graph)
                
                # Calculate isolated nodes penalty if enabled
                isolated_nodes_penalty = 0.0
                if isolated_nodes_penalty_weight > 0:
                    isolated_nodes_penalty = calculate_isolated_nodes_penalty(soft_samples, num_nodes_per_graph)
                    epoch_isolated_penalty += isolated_nodes_penalty.item()
                
                # Calculate triangle penalty if enabled
                triangle_penalty = 0.0
                if triangle_penalty_weight > 0:
                    triangle_penalty = calculate_triangle_penalty(soft_samples, num_nodes_per_graph)
                    epoch_triangle_penalty += triangle_penalty.item()
                
                # Adaptive penalty weights
                adaptive_penalty_weight = degree_penalty_weight * min(2.0, (1 + 0.2 * epoch / epochs))
                
                # Final loss with all penalties
                loss_with_penalty = loss + adaptive_penalty_weight * deg_penalty_diff
                
                if isolated_nodes_penalty_weight > 0:
                    loss_with_penalty = loss_with_penalty + isolated_nodes_penalty_weight * isolated_nodes_penalty
                
                if triangle_penalty_weight > 0:
                    loss_with_penalty = loss_with_penalty + triangle_penalty_weight * triangle_penalty
                
                # Add L2 regularization
                decoder_params = list(model.decoder.decoder_net.parameters())
                l2_reg = 1e-5 * sum(p.pow(2.0).sum() for p in decoder_params)
                loss_with_penalty = loss_with_penalty + l2_reg
                
                loss_with_penalty.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss_with_penalty.item() * batch.num_graphs

            # Calculate epoch statistics
            avg_loss = total_loss / len(dataloader.dataset)
            avg_deg_penalty = epoch_deg_penalty / len(dataloader)
            avg_recon_loss = epoch_recon_loss / len(dataloader)
            avg_kl_loss = epoch_kl_loss / len(dataloader)
            avg_isolated_penalty = epoch_isolated_penalty / len(dataloader) if isolated_nodes_penalty_weight > 0 else 0
            avg_triangle_penalty = epoch_triangle_penalty / len(dataloader) if triangle_penalty_weight > 0 else 0
            
            loss_history.append(avg_loss)
            degree_penalties.append(avg_deg_penalty)
            kl_values.append(avg_kl_loss)
            if isolated_nodes_penalty_weight > 0:
                isolated_node_penalties.append(avg_isolated_penalty)
            if triangle_penalty_weight > 0:
                triangle_penalties.append(avg_triangle_penalty)
            avg_kl_per_epoch.append((avg_kl_loss, kl_weight))
            
            scheduler.step(avg_loss)

            # Check for early stopping
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                no_improvement_count = 0
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

            # Update progress bar
            pbar.set_postfix({'Epoch': epoch, 'Loss': f'{avg_loss:.4f}'})
            
            # Print training stats with additional penalty information
            stats_str = (
                f"Epoch {epoch:3d} | "
                f"Loss: {avg_loss:.4f} | "
                f"Best: {best_loss:.4f} | "
                f"Recon: {avg_recon_loss:.4f} | "
                f"KL: {avg_kl_loss:.4f} (w={kl_weight:.2f}) | "
                f"Deg: {avg_deg_penalty:.4f}"
            )
            
            if isolated_nodes_penalty_weight > 0:
                stats_str += f" | Isol: {avg_isolated_penalty:.4f}"
            
            if triangle_penalty_weight > 0:
                stats_str += f" | Tri: {avg_triangle_penalty:.4f}"
                
            stats_str += f" | Pat: {no_improvement_count}/{patience} | LR: {optimizer.param_groups[0]['lr']:.1e}"
            
            tqdm.write(stats_str)

            # Sample and visualize graphs at regular intervals
            if epoch % 20 == 0 or epoch == epochs:
                model.eval()
                with torch.no_grad():
                    # Sample some graphs to visualize
                    n_samples = 3
                    train_samples = []
                    raw_samples = []
                    
                    for _ in range(n_samples):
                        n = model.node_dist.sample(1)[0]
                        z = model.prior().sample(torch.Size([1]))
                        
                        adj_dist = model.decoder(z).base_dist
                        raw_adj = adj_dist.sample()
                        raw_adj = raw_adj[0, :n, :n]
                        raw_adj.fill_diagonal_(0)
                        raw_samples.append(raw_adj)
                        
                        # Extract largest connected component
                        A_np = raw_adj.detach().cpu().numpy()
                        G = nx.from_numpy_array(A_np)
                        if len(G.nodes()) > 0:
                            largest_cc = max(nx.connected_components(G), key=len)
                            largest_cc = sorted(largest_cc)
                            lcc_adj = raw_adj[np.ix_(largest_cc, largest_cc)]
                            train_samples.append(lcc_adj)
                        else:
                            train_samples.append(raw_adj)  # Empty graph
                    
                    # Visualize the graphs
                    analyze_and_compare_graphs(raw_samples, train_samples, epoch, calculate_degree_penalty)
                
                # Calculate and log active units
                active_units, _ = calculate_active_units(model, dataloader, device)
                active_units_per_epoch.append((epoch, active_units))
                active_percent = 100 * active_units / latent_dim
                print(f"Active dimensions: {active_units}/{latent_dim} ({active_percent:.1f}%)")

    # After training, plot loss curves
    plot_training_loss(loss_history, degree_penalties, kl_values, loss_plot_path)
    print(f"📉 Saved loss curve to {loss_plot_path}")
    
    # Compare training vs generated graph statistics
    print("\n📊 Comparing statistics between training and generated graphs...")
    
    # Get training graphs as PyG Data objects
    training_graphs = []
    for data in dataloader.dataset:
        training_graphs.append(data)
    
    # Generate comparable graphs
    num_generated = min(len(training_graphs), 100)
    model.eval()
    generated_graphs = []
    
    with torch.no_grad():
        for _ in range(num_generated):
            sampled_graph = model.sample(1)[0]
            generated_graphs.append(sampled_graph)
    
    # Compute statistics

    train_degrees, train_clustering, _ = compute_graph_statistics(training_graphs)
    gen_degrees, gen_clustering, _ = compute_graph_statistics(generated_graphs)
    
    # Print summary statistics
    train_clustering_mean = np.mean(train_clustering) if train_clustering else 0
    train_clustering_std = np.std(train_clustering) if train_clustering else 0
    gen_clustering_mean = np.mean(gen_clustering) if gen_clustering else 0
    gen_clustering_std = np.std(gen_clustering) if gen_clustering else 0
    
    print("\n===== Clustering Coefficient Statistics =====")
    print(f"Training graphs: mean={train_clustering_mean:.4f}, std={train_clustering_std:.4f}")
    print(f"Generated graphs: mean={gen_clustering_mean:.4f}, std={gen_clustering_std:.4f}")
    print(f"Difference in means: {abs(train_clustering_mean - gen_clustering_mean):.4f}")
    
    print("\n===== Node Degree Statistics =====")
    train_degree_mean = np.mean(train_degrees) if train_degrees else 0
    train_degree_std = np.std(train_degrees) if train_degrees else 0
    gen_degree_mean = np.mean(gen_degrees) if gen_degrees else 0
    gen_degree_std = np.std(gen_degrees) if gen_degrees else 0
    
    print(f"Training graphs: mean={train_degree_mean:.4f}, std={train_degree_std:.4f}")
    print(f"Generated graphs: mean={gen_degree_mean:.4f}, std={gen_degree_std:.4f}")
    print(f"Difference in means: {abs(train_degree_mean - gen_degree_mean):.4f}")


import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj
import random
from collections import defaultdict
import queue

# Disable interactive plotting
plt.ioff()

# Import our Graph VAE model
from graph_vae import GraphVAE, load_data, create_dataloaders

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pyg_to_networkx(data):
    """Convert a PyTorch Geometric data object to a NetworkX graph"""
    G = nx.Graph()
    
    # Add nodes
    for i in range(data.num_nodes):
        G.add_node(i)
    
    # Add edges
    edge_list = data.edge_index.t().tolist()
    G.add_edges_from(edge_list)
    
    return G

def sample_graphs_from_vae(model, num_samples, max_nodes, original_node_counts=None):
    """
    Sample graphs from the VAE using our custom degree-constrained sampling method.
    
    Args:
        model: The trained GraphVAE model
        num_samples: Number of graphs to sample
        max_nodes: Maximum number of nodes allowed
        original_node_counts: List of node counts from original dataset to sample from
    """
    # Sample node counts from the original distribution
    if original_node_counts:
        # Randomly sample node counts from the original distribution
        sampled_node_counts = random.choices(original_node_counts, k=num_samples)
    else:
        # Fallback to random counts if not provided
        sampled_node_counts = [random.randint(10, max_nodes) for _ in range(num_samples)]
        
    # Generate graphs with sampled node counts
    node_features, adj_sampled, node_mask = model.sample(num_samples, sampled_node_counts)
    graphs = []

    for i in range(num_samples):
        # Extract sampled adjacency matrix and apply node mask
        adj_matrix = adj_sampled[i].cpu().numpy()
        mask = node_mask[i].bool().cpu().numpy()
        actual_nodes = int(mask.sum())
        
        # Create NetworkX graph with only the active nodes
        G = nx.Graph()
        
        # Add nodes
        for j in range(actual_nodes):
            G.add_node(j)
            
        # Add edges between active nodes
        for j in range(actual_nodes):
            for k in range(j+1, actual_nodes):
                if adj_matrix[j, k] > 0.5:  # Edge exists
                    G.add_edge(j, k)
        
        # Remove isolated nodes (degree 0) if any
        G.remove_nodes_from(list(nx.isolates(G)))
        
        graphs.append(G)
        
    return graphs

def compute_metrics(graphs):
    """
    Compute degree, clustering, and eigenvector centrality for a list of graphs.
    """
    degrees, clustering, eigenvector = [], [], []
    num_nodes_list = []
    disconnected_count = 0
    for G in graphs:
        if G.number_of_nodes() == 0:
            continue
        num_nodes_list.append(G.number_of_nodes())
        degs = [d for _, d in G.degree()]
        degrees.extend(degs)
        try:
            clust = nx.clustering(G)
            clustering.extend(clust.values())
        except nx.NetworkXError:
            clustering.extend([0.0] * G.number_of_nodes())

        try:
            # Increase max_iter and tol for robustness
            eig = nx.eigenvector_centrality_numpy(G)
            eigenvector.extend(eig.values())
        except (nx.NetworkXError, nx.PowerIterationFailedConvergence, ZeroDivisionError, nx.AmbiguousSolution):
            # Handle convergence issues, graphs where centrality is zero, or disconnected graphs
            eigenvector.extend([0.0] * G.number_of_nodes())

        if G.number_of_nodes() > 0 and not nx.is_connected(G):
             disconnected_count += 1

    return degrees, clustering, eigenvector, num_nodes_list, disconnected_count

def plot_histograms(original_metrics, vae_metrics, filename='method_comparison_v2.png'):
    """
    Plot histograms for degree, clustering, and eigenvector centrality.
    """
    method_names = ['Original', 'Graph VAE']
    metric_names = ['Degree', 'Clustering', 'Eigenvector']
    all_metrics = [original_metrics[:3], vae_metrics[:3]]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            data = all_metrics[i][j]
            if not data:
                ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                continue

            if j == 0:
                max_deg = int(np.max(data)) if data else 0
                bins = np.arange(0, max_deg + 2) - 0.5
            else:
                bins = np.linspace(min(data) if data else 0, max(data) if data else 1, 21)

            ax.hist(data, bins=bins, density=True, alpha=0.7, label=method_names[i])
            ax.set_title(metric_names[j])
            ax.set_xlabel('Value')
            ax.set_ylabel('Density' if j == 0 else '')
            ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved histogram comparison to {filename}")

def plot_sample_graphs(original_graphs, vae_graphs, filename='sample_graphs_comparison_v2.png'):
    """
    Plot sample graphs from original dataset and VAE.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, graphs in enumerate([original_graphs, vae_graphs]):
        num_to_sample = min(3, len(graphs))
        if num_to_sample == 0:
            for j in range(3):
                ax = axes[i, j]
                ax.text(0.5, 0.5, 'No Graphs', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{'Original' if i == 0 else 'VAE'} Sample {j+1}")
            continue

        samples = random.sample(graphs, num_to_sample)
        for j in range(3):
            ax = axes[i, j]
            if j < len(samples):
                G = samples[j]
                pos = nx.spring_layout(G, seed=42) if G.number_of_nodes() > 0 else {}
                nx.draw(G, pos, node_size=50, ax=ax, width=0.5)
                title = 'Original' if i == 0 else 'VAE'
                ax.set_title(f"{title} Sample {j+1} (N={G.number_of_nodes()})")
            else:
                 ax.set_title(f"{'Original' if i == 0 else 'VAE'} Sample {j+1}")
                 ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved sample graph comparison to {filename}")

def print_statistics(original_graphs, vae_graphs, original_metrics, vae_metrics):
    """
    Print summary and detailed degree statistics.
    """
    print("\nSummary Statistics:")
    metrics_list = [original_metrics, vae_metrics]
    names = ['Original', 'Graph VAE']

    for i, (name, metrics) in enumerate(zip(names, metrics_list)):
        degs, clustering, eigen, num_nodes_list, disconnected_count = metrics
        graphs = original_graphs if i == 0 else vae_graphs
        print(f"\n{name}")
        if not degs:
            print("  No valid graphs generated/found.")
            continue

        print(f"  Num Graphs: {len(graphs)}")
        print(f"  Avg Nodes: {np.mean(num_nodes_list):.2f}")
        print(f"  Node Count Min: {min(num_nodes_list) if num_nodes_list else 0}")
        print(f"  Node Count Max: {max(num_nodes_list) if num_nodes_list else 0}")
        print(f"  Node Count Std Dev: {np.std(num_nodes_list):.2f}")
        print(f"  Mean Degree: {np.mean(degs):.2f}")
        nonzero_degs = [d for d in degs if d > 0]
        print(f"  Mean Nonzero Degree: {np.mean(nonzero_degs) if nonzero_degs else 0:.2f}")
        print(f"  Clustering: {np.mean(clustering):.2f}")
        print(f"  Eigenvector: {np.mean(eigen):.2f}")
        if name == 'Graph VAE':
            print(f"  Disconnected graphs: {disconnected_count}/{len(graphs)} ({disconnected_count/len(graphs)*100:.1f}%)")

    print("\nDegree Distribution (Original vs VAE):")
    max_degree_to_show = 5
    for i, (name, metrics) in enumerate(zip(names, metrics_list)):
        degs = metrics[0]
        print(f"\n{name}:")
        if not degs:
            print("  No degree data.")
            continue

        total_nodes = len(degs)
        counts = defaultdict(int)
        for d in degs:
            counts[d] += 1

        max_observed_degree = max(counts.keys()) if counts else -1

        for k in range(max_degree_to_show + 1):
            cnt = counts[k]
            print(f"  Degree {k}: {cnt} ({cnt/total_nodes*100:.1f}%)")

        cnt_gt = sum(counts[d] for d in counts if d > max_degree_to_show)
        if cnt_gt > 0:
            print(f"  Degree >{max_degree_to_show}: {cnt_gt} ({cnt_gt/total_nodes*100:.1f}%)")
        elif max_observed_degree > max_degree_to_show:
             print(f"  Degree >{max_degree_to_show}: 0 (0.0%)")

if __name__ == '__main__':
    # --- Config (should match training config) ---
    config = {
        'hidden_dim': 64,
        'latent_dim': 32,
        'num_layers': 3,
        'model_load_path': 'graph_vae_model.pt'
    }

    # --- Load Data ---
    dataset, node_feature_dim, max_nodes, _, original_node_counts = load_data()  # Updated to unpack 5 values
    original_graphs = [pyg_to_networkx(d) for d in dataset]
    print(f"Loaded {len(original_graphs)} original graphs. Max nodes: {max_nodes}")
    print(f"Original node count statistics: Mean: {sum(original_node_counts)/len(original_node_counts):.2f}, Min: {min(original_node_counts)}, Max: {max(original_node_counts)}")

    # --- Load Model ---
    model = GraphVAE(node_feature_dim, config['hidden_dim'], config['latent_dim'], max_nodes, config['num_layers']).to(device)
    try:
        model.load_state_dict(torch.load(config['model_load_path'], map_location=device, weights_only=True))
        model.eval()
        print(f"Loaded trained model from {config['model_load_path']}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {config['model_load_path']}. Cannot perform comparison.")
        exit()
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure the model architecture in graph_vae.py matches the saved model.")
        exit()

    # --- Sample and Evaluate ---
    print(f"Sampling {len(original_graphs)} graphs from VAE...")
    vae_graphs = sample_graphs_from_vae(model, len(original_graphs), max_nodes, original_node_counts)
    print("Computing metrics...")
    orig_metrics = compute_metrics(original_graphs)
    vae_metrics = compute_metrics(vae_graphs)

    # --- Output Results ---
    plot_histograms(orig_metrics, vae_metrics, filename='method_comparison_v3.png') # Save v3 plot
    plot_sample_graphs(original_graphs, vae_graphs, filename='sample_graphs_comparison_v3.png') # Save v3 plot
    print_statistics(original_graphs, vae_graphs, orig_metrics, vae_metrics)
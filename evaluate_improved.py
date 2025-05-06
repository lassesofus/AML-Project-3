import torch
import numpy as np
import random
import networkx as nx
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split

from src.utils.data import load_data
from src.utils.graph_utils import pyg_to_networkx
from src.utils.plot import plot_histograms, plot_sample_graphs
from src.models.improved_model import create_improved_model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def sample_graphs_from_improved_model(model, num_samples, max_nodes, original_node_counts=None):
    """
    Sample graphs from the improved Graph VAE model.
    
    Args:
        model: The trained ImprovedGraphVAE model
        num_samples: Number of graphs to sample
        max_nodes: Maximum number of nodes allowed
        original_node_counts: List of node counts from original dataset to sample from
    
    Returns:
        graphs: List of NetworkX graphs
    """
    # Sample node counts from the original distribution
    if original_node_counts:
        # Create sizes tensor based on original distribution
        sizes = torch.tensor([random.choice(original_node_counts) for _ in range(num_samples)])
    else:
        # Fallback to random counts if not provided
        sizes = torch.tensor([random.randint(10, max_nodes) for _ in range(num_samples)])
    
    # Generate adjacency matrices
    with torch.no_grad():
        adj_sampled = model.sample(num_samples, sizes=sizes)
    
    graphs = []
    total_edges = 0
    total_nodes = 0
    
    print(f"\nDebug - Graph Generation Statistics:")
    print(f"Sampled {num_samples} graphs with these properties:")
    
    for i in range(num_samples):
        # Extract sampled adjacency matrix
        adj_matrix = adj_sampled[i].cpu().numpy()
        
        # Determine actual number of nodes (non-zero rows/columns)
        mask = (adj_matrix.sum(axis=1) + adj_matrix.sum(axis=0)) > 0
        actual_nodes = int(mask.sum())
        if actual_nodes == 0:
            # If no active nodes detected, use the requested size
            actual_nodes = min(sizes[i].item() if i < len(sizes) else max_nodes, max_nodes)
        
        total_nodes += actual_nodes
        
        # Create NetworkX graph with only the active nodes
        G = nx.Graph()
        
        # Add nodes
        for j in range(actual_nodes):
            G.add_node(j)
        
        # Add edges between active nodes
        edge_count = 0
        for j in range(actual_nodes):
            for k in range(j+1, actual_nodes):  # Only lower triangular part for undirected graph
                if adj_matrix[j, k] > 0.5:  # Threshold for edge creation
                    G.add_edge(j, k)
                    edge_count += 1
        
        total_edges += edge_count
        
        # Add the graph
        graphs.append(G)
        
        # Debug every 20th graph or if it has unusual properties
        if i % 20 == 0 or edge_count == 0 or actual_nodes < 3:
            is_connected = nx.is_connected(G) if actual_nodes > 0 and edge_count > 0 else "N/A"
            n_components = nx.number_connected_components(G) if actual_nodes > 0 else 0
            print(f"  Graph {i}: Nodes={actual_nodes}, Edges={edge_count}, " 
                  f"Connected={is_connected}, Components={n_components}")
    
    if graphs:
        avg_nodes = total_nodes / len(graphs)
        avg_edges = total_edges / len(graphs)
        print(f"\nSummary: Average nodes per graph: {avg_nodes:.2f}, Average edges per graph: {avg_edges:.2f}")
        
    return graphs

def compute_metrics(graphs):
    """
    Compute degree, clustering, and eigenvector centrality for a list of graphs.
    """
    degrees, clustering, eigenvector = [], [], []
    num_nodes_list = []
    disconnected_count = 0
    
    print("\nMetrics Calculation:")
    total_components = 0
    total_isolated = 0
    
    for i, G in enumerate(graphs):
        if G.number_of_nodes() == 0:
            continue
            
        # Track full graph statistics
        num_nodes_list.append(G.number_of_nodes())
        n_components = nx.number_connected_components(G)
        total_components += n_components
        isolated_nodes = list(nx.isolates(G))
        total_isolated += len(isolated_nodes)
        
        # Find largest connected component for metrics
        if not nx.is_connected(G):
            disconnected_count += 1
            components = list(nx.connected_components(G))
            largest_cc = max(components, key=len)
            # Get subgraph of largest component
            G_lcc = G.subgraph(largest_cc).copy()
            
            if i % 50 == 0:
                print(f"  Graph {i}: Full size={G.number_of_nodes()}, Components={n_components}, " 
                      f"LCC size={G_lcc.number_of_nodes()}, Isolated nodes={len(isolated_nodes)}")
        else:
            G_lcc = G
            
        # Only compute metrics if LCC has nodes
        if G_lcc.number_of_nodes() == 0:
            continue
            
        # Compute degrees on LCC
        degs = [d for _, d in G_lcc.degree()]
        degrees.extend(degs)
        
        # Compute clustering coefficients on LCC
        if G_lcc.number_of_nodes() >= 3 and G_lcc.number_of_edges() > 0:
            clust = nx.clustering(G_lcc)
            clustering.extend(clust.values())
        else:
            clustering.extend([0.0] * G_lcc.number_of_nodes())

        # Compute eigenvector centrality on LCC
        if G_lcc.number_of_nodes() < 3 or G_lcc.number_of_edges() < 2:
            # For very small graphs, assign equal centrality
            eig_vals = [1.0/G_lcc.number_of_nodes()] * G_lcc.number_of_nodes()
            eigenvector.extend(eig_vals)
        else:
            # Standard eigenvector centrality calculation
            try:
                eig = nx.eigenvector_centrality(G_lcc, max_iter=1000, tol=1.0e-6)
                eigenvector.extend(eig.values())
            except:
                # Fallback to degree centrality if eigenvector computation fails
                eig = nx.degree_centrality(G_lcc)
                eigenvector.extend(eig.values())

    # Print summary statistics
    if graphs:
        print(f"\nMetrics Summary:")
        print(f"  Total graphs: {len(graphs)}")
        print(f"  Disconnected graphs: {disconnected_count} ({disconnected_count/len(graphs)*100:.1f}%)")
        print(f"  Average components per graph: {total_components/len(graphs):.2f}")
        print(f"  Total isolated nodes: {total_isolated}")

    return degrees, clustering, eigenvector, num_nodes_list, disconnected_count

def print_statistics(original_graphs, vae_graphs, original_metrics, vae_metrics):
    """
    Print summary and detailed degree statistics.
    """
    print("\nSummary Statistics:")
    metrics_list = [original_metrics, vae_metrics]
    names = ['Target (Original)', 'Improved Graph VAE']

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
        
        if name == 'Improved Graph VAE':
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

def plot_custom_histograms(orig_metrics, vae_metrics, filename='improved_histograms.png'):
    """
    Plot histograms comparing original and generated graph properties.
    """
    orig_degrees, orig_clustering, orig_eigenvector, orig_nodes, _ = orig_metrics
    vae_degrees, vae_clustering, vae_eigenvector, vae_nodes, _ = vae_metrics
    
    plt.figure(figsize=(15, 5))
    
    # Degree distribution
    plt.subplot(1, 3, 1)
    max_degree = max(max(orig_degrees, default=0), max(vae_degrees, default=0))
    bins = list(range(max_degree + 2))  # +2 to include max_degree in a bin
    plt.hist([orig_degrees, vae_degrees], bins=bins, alpha=0.7, label=['Original', 'Improved VAE'])
    plt.xlabel('Node Degree')
    plt.ylabel('Count')
    plt.title('Degree Distribution')
    plt.legend()
    
    # Clustering coefficient
    plt.subplot(1, 3, 2)
    plt.hist([orig_clustering, vae_clustering], bins=10, alpha=0.7, label=['Original', 'Improved VAE'])
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Count')
    plt.title('Clustering Coefficient Distribution')
    plt.legend()
    
    # Eigenvector centrality
    plt.subplot(1, 3, 3)
    plt.hist([orig_eigenvector, vae_eigenvector], bins=10, alpha=0.7, label=['Original', 'Improved VAE'])
    plt.xlabel('Eigenvector Centrality')
    plt.ylabel('Count')
    plt.title('Eigenvector Centrality Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_custom_samples(original_graphs, generated_graphs, filename='improved_samples.png'):
    """
    Plot sample original and generated graphs for visual comparison.
    """
    plt.figure(figsize=(15, 10))
    
    # Select a few graphs to display (3 original, 3 generated)
    num_samples = min(3, len(original_graphs), len(generated_graphs))
    
    # Randomly select indices
    random.seed(42)  # For reproducibility
    if len(original_graphs) > num_samples:
        orig_indices = random.sample(range(len(original_graphs)), num_samples)
    else:
        orig_indices = range(len(original_graphs))
        
    if len(generated_graphs) > num_samples:
        gen_indices = random.sample(range(len(generated_graphs)), num_samples)
    else:
        gen_indices = range(len(generated_graphs))
    
    # Plot original graphs
    for i, idx in enumerate(orig_indices):
        plt.subplot(2, num_samples, i+1)
        G = original_graphs[idx]
        nx.draw(G, node_size=50, alpha=0.8)
        plt.title(f"Original Graph {idx}\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Plot generated graphs
    for i, idx in enumerate(gen_indices):
        plt.subplot(2, num_samples, num_samples + i+1)
        G = generated_graphs[idx]
        nx.draw(G, node_size=50, alpha=0.8)
        plt.title(f"Generated Graph {idx}\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def evaluate_improved(model_path='graph_vae_model_improved.pt', output_prefix='improved'):
    """
    Evaluate our improved Graph VAE model.
    """
    # Load dataset
    dataset = TUDataset(root='./data/', name='MUTAG')
    node_feature_dim = dataset.num_node_features
    original_node_counts = [data.num_nodes for data in dataset]
    max_nodes = max(original_node_counts)
    
    # Convert dataset to NetworkX graphs for evaluation
    original_graphs = []
    for data in dataset:
        G = nx.Graph()
        
        # Add nodes
        for i in range(data.num_nodes):
            G.add_node(i)
        
        # Add edges from edge_index
        for i in range(data.edge_index.shape[1]):
            src, dst = data.edge_index[:, i].tolist()
            G.add_edge(src, dst)
        
        original_graphs.append(G)
    
    print(f"Loaded {len(original_graphs)} original graphs. Max nodes: {max_nodes}")
    print(f"Original node count statistics: Mean: {np.mean(original_node_counts):.2f}, " 
          f"Min: {min(original_node_counts)}, Max: {max(original_node_counts)}")
    
    # First attempt to load the model to determine its architecture
    temp_model = torch.load(model_path, map_location=device)
    
    # Check if it's a state_dict or full model
    if isinstance(temp_model, dict) and 'encoder.message_net.3.0.weight' in temp_model:
        print("Detected model with 4 message passing rounds")
        num_message_passing_rounds = 4
    else:
        print("Using default 3 message passing rounds")
        num_message_passing_rounds = 3
    
    # Check hidden dimension by examining decoder network parameters
    if isinstance(temp_model, dict) and 'decoder.decoder_net.0.weight' in temp_model:
        # Get output dimension of first layer (hidden_dim * 4)
        hidden_dim_times_4 = temp_model['decoder.decoder_net.0.weight'].size(0)
        hidden_dim = hidden_dim_times_4 // 4
        print(f"Detected hidden dimension: {hidden_dim}")
    else:
        print("Using default hidden dimension: 8")
        hidden_dim = 8
    
    # Determine latent dimension
    if isinstance(temp_model, dict) and 'decoder.decoder_net.0.weight' in temp_model:
        latent_dim = temp_model['decoder.decoder_net.0.weight'].size(1)
        print(f"Detected latent dimension: {latent_dim}")
    else:
        print("Using default latent dimension: 5")
        latent_dim = 5
    
    # Create our improved model with detected architecture
    model = create_improved_model(
        node_feature_dim=node_feature_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        max_nodes=max_nodes,
        num_message_passing_rounds=num_message_passing_rounds,
        encoder_type='mp'
    ).to(device)
    
    # Load trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Successfully loaded model from {model_path}")
    
    # Set up graph size sampling function
    def sample_size_fn(n_samples):
        return np.random.choice(original_node_counts, size=n_samples)
    
    model.set_sample_size_function(sample_size_fn)
    
    # Sample and evaluate
    num_samples = len(original_graphs)
    print(f"Sampling {num_samples} graphs from improved VAE...")
    vae_graphs = sample_graphs_from_improved_model(model, num_samples, max_nodes, original_node_counts)
    
    print("Computing metrics...")
    orig_metrics = compute_metrics(original_graphs)
    vae_metrics = compute_metrics(vae_graphs)
    
    # Generate output
    plot_custom_histograms(orig_metrics, vae_metrics, filename=f'figures/{output_prefix}_histograms.png')
    plot_custom_samples(original_graphs, vae_graphs, filename=f'figures/{output_prefix}_samples.png')
    print_statistics(original_graphs, vae_graphs, orig_metrics, vae_metrics)
    
    return original_graphs, vae_graphs

if __name__ == '__main__':
    evaluate_improved()
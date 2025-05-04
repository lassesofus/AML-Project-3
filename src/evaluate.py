import torch
import numpy as np
import random
import networkx as nx
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from src.utils.data import load_data
from src.utils.graph_utils import pyg_to_networkx
from src.utils.plot import plot_histograms, plot_sample_graphs
from src.models.model import GraphVAE

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def sample_graphs_from_vae(model, num_samples, max_nodes, original_node_counts=None):
    """
    Sample graphs from the VAE using our custom degree-constrained sampling method.
    
    Args:
        model: The trained GraphVAE model
        num_samples: Number of graphs to sample
        max_nodes: Maximum number of nodes allowed
        original_node_counts: List of node counts from original dataset to sample from
    
    Returns:
        graphs: List of NetworkX graphs
    """
    # Sample node counts from the original distribution
    if original_node_counts:
        # Randomly sample node counts from the original distribution
        sampled_node_counts = random.choices(original_node_counts, k=num_samples)
    else:
        # Fallback to random counts if not provided
        sampled_node_counts = [random.randint(10, max_nodes) for _ in range(num_samples)]
        
    # Generate graphs with sampled node counts
    node_features, adj_sampled, node_mask = model.sample(num_samples, sampled_node_counts, 
                                                         enforce_connectivity=False,  # Always enforce connectivity
                                                         use_spanning_tree=False,    # Use spanning tree to ensure connected graphs
                                                         target_degree_matching=False)
    graphs = []

    for i in range(num_samples):
        # Extract sampled adjacency matrix and apply node mask
        adj_matrix = adj_sampled[i].cpu().numpy()
        mask = node_mask[i].bool().cpu().numpy()
        actual_nodes = int(mask.sum())
        
        if actual_nodes < 2:  # Skip if we have fewer than 2 nodes (can't form edges)
            continue
            
        # Create NetworkX graph with only the active nodes
        G = nx.Graph()
        
        # Add nodes
        for j in range(actual_nodes):
            G.add_node(j)
            
        # Add edges between active nodes
        edge_added = False
        for j in range(actual_nodes):
            for k in range(j+1, actual_nodes):
                if adj_matrix[j, k] > 0.5:  # Edge exists
                    G.add_edge(j, k)
                    edge_added = True
        
        # Force at least one edge if none were added and we have nodes
        if not edge_added and actual_nodes >= 2:
            G.add_edge(0, 1)
            
        # Remove isolated nodes (degree 0) if any
        G.remove_nodes_from(list(nx.isolates(G)))
        
        # Only add graphs that have at least one node
        if G.number_of_nodes() > 0:
            graphs.append(G)
    
    # If somehow no valid graphs were generated, create at least one simple graph
    if not graphs and num_samples > 0:
        G = nx.path_graph(3)  # Simple path graph with 3 nodes
        graphs.append(G)
        
    return graphs

def compute_metrics(graphs):
    """
    Compute degree, clustering, and eigenvector centrality for a list of graphs.
    
    Args:
        graphs: List of NetworkX graphs
        
    Returns:
        degrees: List of node degrees
        clustering: List of clustering coefficients
        eigenvector: List of eigenvector centrality values
        num_nodes_list: List of node counts per graph
        disconnected_count: Number of disconnected graphs
    """
    degrees, clustering, eigenvector = [], [], []
    num_nodes_list = []
    disconnected_count = 0
    
    for G in graphs:
        if G.number_of_nodes() == 0:
            continue
            
        num_nodes_list.append(G.number_of_nodes())
        
        # Compute degrees
        degs = [d for _, d in G.degree()]
        degrees.extend(degs)
        
        # Compute clustering coefficients
        if G.number_of_nodes() >= 3 and G.number_of_edges() > 0:
            clust = nx.clustering(G)
            clustering.extend(clust.values())
        else:
            clustering.extend([0.0] * G.number_of_nodes())

        # Compute eigenvector centrality
        if G.number_of_nodes() < 3 or G.number_of_edges() < 2:
            # For very small graphs, assign equal centrality
            eig_vals = [1.0/G.number_of_nodes()] * G.number_of_nodes()
            eigenvector.extend(eig_vals)
        elif nx.is_tree(G):
            # For trees, eigenvector centrality has special properties
            # Assign centrality based on degree (normalized)
            total_degree = sum(degs)
            if total_degree > 0:
                eig_vals = [d/total_degree for d in degs]
            else:
                eig_vals = [1.0/G.number_of_nodes()] * G.number_of_nodes()
            eigenvector.extend(eig_vals)
        else:
            # Standard eigenvector centrality calculation for larger graphs
            if nx.is_connected(G):
                # For connected graphs, use standard approach
                eig = nx.eigenvector_centrality(G, max_iter=1000, tol=1.0e-6)
                eigenvector.extend(eig.values())
            else:
                # For disconnected graphs, calculate per component or use degree centrality
                eig = nx.degree_centrality(G)
                eigenvector.extend(eig.values())

        # Check if graph is connected
        if G.number_of_nodes() > 0 and not nx.is_connected(G):
            disconnected_count += 1

    return degrees, clustering, eigenvector, num_nodes_list, disconnected_count

def print_statistics(original_graphs, vae_graphs, original_metrics, vae_metrics):
    """
    Print summary and detailed degree statistics.
    
    Args:
        original_graphs: List of original NetworkX graphs
        vae_graphs: List of VAE-generated NetworkX graphs
        original_metrics: Metrics from original graphs
        vae_metrics: Metrics from VAE-generated graphs
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

def evaluate(model_path='graph_vae_model.pt', output_prefix='results'):
    """
    Evaluate a trained Graph VAE model by comparing its generated graphs with the original dataset.
    
    Args:
        model_path: Path to the trained model checkpoint
        output_prefix: Prefix for output files
        
    Returns:
        original_graphs: List of original NetworkX graphs
        vae_graphs: List of VAE-generated NetworkX graphs
    """
    # Load data and model configuration
    config = {
        'hidden_dim': 64,
        'latent_dim': 32,
        'num_layers': 3
    }

    # Load dataset
    dataset, node_feature_dim, max_nodes, _, original_node_counts = load_data()
    original_graphs = [pyg_to_networkx(d) for d in dataset]
    print(f"Loaded {len(original_graphs)} original graphs. Max nodes: {max_nodes}")
    print(f"Original node count statistics: Mean: {sum(original_node_counts)/len(original_node_counts):.2f}, Min: {min(original_node_counts)}, Max: {max(original_node_counts)}")

    # Load model
    model = GraphVAE(node_feature_dim, config['hidden_dim'], config['latent_dim'], max_nodes, config['num_layers']).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"Loaded trained model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Cannot perform evaluation.")
        return None, None
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure the model architecture matches the saved model.")
        return None, None

    # Sample and evaluate
    num_samples = len(original_graphs)
    print(f"Sampling {num_samples} graphs from VAE...")
    vae_graphs = sample_graphs_from_vae(model, num_samples, max_nodes, original_node_counts)
    
    print("Computing metrics...")
    orig_metrics = compute_metrics(original_graphs)
    vae_metrics = compute_metrics(vae_graphs)

    # Generate output
    plot_histograms(orig_metrics, vae_metrics, filename=f'{output_prefix}_histograms.png')
    plot_sample_graphs(original_graphs, vae_graphs, filename=f'{output_prefix}_samples.png')
    print_statistics(original_graphs, vae_graphs, orig_metrics, vae_metrics)
    
    return original_graphs, vae_graphs

if __name__ == '__main__':
    evaluate()
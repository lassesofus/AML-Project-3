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
    Sample graphs from the VAE without any post-processing to see the raw model output.
    
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
        
    # Generate graphs with sampled node counts - disable all post-processing
    node_features, adj_sampled, node_mask = model.sample(num_samples, sampled_node_counts)
    graphs = []
    total_edges = 0
    total_nodes = 0
    all_probs = []  # Store all edge probabilities for analysis

    print("\nDebug - Raw Graph Generation Statistics:")
    print(f"Sampled {num_samples} graphs with these properties:")
    
    # Set a lower threshold for edge creation to see if we get any edges
    edge_threshold = 0.1  # Try a much lower threshold than 0.5
    print(f"Using edge probability threshold: {edge_threshold}")

    for i in range(num_samples):
        # Extract sampled adjacency matrix and apply node mask
        adj_matrix = adj_sampled[i].cpu().numpy()
        mask = node_mask[i].bool().cpu().numpy()
        actual_nodes = int(mask.sum())
        total_nodes += actual_nodes
        
        # Store edge probabilities for the first few graphs to analyze the distribution
        if i < 5:
            for j in range(actual_nodes):
                for k in range(j+1, actual_nodes):
                    all_probs.append(adj_matrix[j, k])
        
        # Create NetworkX graph with only the active nodes
        G = nx.Graph()
        
        # Add nodes
        for j in range(actual_nodes):
            G.add_node(j)
            
        # Add edges between active nodes - using lower threshold
        edge_count = 0
        max_prob = 0
        for j in range(actual_nodes):
            for k in range(j+1, actual_nodes):
                prob = adj_matrix[j, k]
                max_prob = max(max_prob, prob)
                if prob > edge_threshold:  # Edge exists with lower threshold
                    G.add_edge(j, k)
                    edge_count += 1
        
        total_edges += edge_count
        
        # No post-processing - add the graph even if it has no edges or isolated nodes
        graphs.append(G)
        
        # Debug every 20th graph or if it has unusual properties
        if i % 20 == 0 or edge_count == 0 or actual_nodes < 3:
            print(f"  Graph {i}: Nodes={actual_nodes}, Edges={edge_count}, " 
                  f"Max Prob={max_prob:.4f}, Connected={nx.is_connected(G) if actual_nodes > 0 else 'N/A'}, "
                  f"Components={nx.number_connected_components(G) if actual_nodes > 0 else 0}")
    
    if graphs:
        avg_nodes = total_nodes / len(graphs)
        avg_edges = total_edges / len(graphs)
        print(f"\nSummary: Average nodes per graph: {avg_nodes:.2f}, Average edges per graph: {avg_edges:.2f}")
        
    # Analyze edge probability distribution
    if all_probs:
        all_probs = np.array(all_probs)
        print(f"\nEdge probability statistics (from first 5 graphs):")
        print(f"  Min: {all_probs.min():.6f}, Max: {all_probs.max():.6f}")
        print(f"  Mean: {all_probs.mean():.6f}, Median: {np.median(all_probs):.6f}")
        print(f"  5th percentile: {np.percentile(all_probs, 5):.6f}")
        print(f"  95th percentile: {np.percentile(all_probs, 95):.6f}")
        
        # Count probabilities in ranges
        ranges = [(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 1.0)]
        for low, high in ranges:
            count = ((all_probs >= low) & (all_probs < high)).sum()
            percent = count / len(all_probs) * 100
            print(f"  Probability [{low:.2f}-{high:.2f}): {count} values ({percent:.1f}%)")
        
    return graphs

def compute_metrics(graphs):
    """
    Compute degree, clustering, and eigenvector centrality for a list of graphs.
    For each graph, metrics are calculated on the largest connected component
    while overall statistics track the full graphs.
    
    Args:
        graphs: List of NetworkX graphs
        
    Returns:
        degrees: List of node degrees from largest connected components
        clustering: List of clustering coefficients from largest connected components
        eigenvector: List of eigenvector centrality values from largest connected components
        num_nodes_list: List of node counts per graph (full graphs)
        disconnected_count: Number of disconnected graphs
    """
    degrees, clustering, eigenvector = [], [], []
    num_nodes_list = []
    disconnected_count = 0
    
    print("\nDebug - Metrics Calculation:")
    print("Using largest connected component of each graph for metrics")
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
            
            # Debug output for some graphs
            if i % 50 == 0 or G.number_of_nodes() < 3:
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
        elif nx.is_tree(G_lcc):
            # For trees, use degree centrality (normalized)
            total_degree = sum(degs)
            if total_degree > 0:
                eig_vals = [d/total_degree for d in degs]
            else:
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
    
    Args:
        original_graphs: List of original NetworkX graphs
        vae_graphs: List of VAE-generated NetworkX graphs
        original_metrics: Metrics from original graphs
        vae_metrics: Metrics from VAE-generated graphs
    """
    print("\nSummary Statistics:")
    metrics_list = [original_metrics, vae_metrics]
    names = ['Target (Original)', 'Graph VAE']

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
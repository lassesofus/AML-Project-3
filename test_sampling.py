import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from src.models.model import GraphVAE
from src.utils.data import load_data
from collections import Counter

# For reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_degree_distribution(original_graphs, generated_graphs, filename='degree_distribution_test.png'):
    """Plot degree distributions of original and generated graphs"""
    # Count degrees
    orig_degrees = []
    for G in original_graphs:
        orig_degrees.extend([d for _, d in G.degree()])
    
    gen_degrees = []
    for G in generated_graphs:
        gen_degrees.extend([d for _, d in G.degree()])
    
    # Count degree occurrences
    orig_counter = Counter(orig_degrees)
    gen_counter = Counter(gen_degrees)
    
    # Calculate percentages
    total_orig = len(orig_degrees)
    total_gen = len(gen_degrees)
    
    max_degree = max(max(orig_counter.keys()), max(gen_counter.keys()))
    
    plt.figure(figsize=(12, 6))
    
    # Original graph degrees
    orig_pcts = [orig_counter.get(i, 0) / total_orig * 100 for i in range(max_degree + 1)]
    plt.bar(np.arange(max_degree + 1) - 0.2, orig_pcts, width=0.4, label='Original')
    
    # Generated graph degrees
    gen_pcts = [gen_counter.get(i, 0) / total_gen * 100 for i in range(max_degree + 1)]
    plt.bar(np.arange(max_degree + 1) + 0.2, gen_pcts, width=0.4, label='Generated')
    
    plt.xlabel('Node Degree')
    plt.ylabel('Percentage of Nodes (%)')
    plt.title('Degree Distribution Comparison')
    plt.xticks(range(max_degree + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Print statistics
    print("\nDegree Distribution (Original vs Generated):")
    for i in range(max_degree + 1):
        orig_pct = orig_counter.get(i, 0) / total_orig * 100
        gen_pct = gen_counter.get(i, 0) / total_gen * 100
        print(f"  Degree {i}: {orig_pct:.1f}% vs {gen_pct:.1f}%")
    
    plt.savefig(filename, dpi=300)
    print(f"Saved degree distribution plot to {filename}")

def convert_to_networkx(node_features, adj_matrix, node_mask):
    """Convert model output to NetworkX graph"""
    graphs = []
    batch_size = adj_matrix.shape[0]
    
    for b in range(batch_size):
        G = nx.Graph()
        mask = node_mask[b].bool()
        n_nodes = mask.sum().item()
        
        # Add nodes
        for i in range(n_nodes):
            G.add_node(i)
        
        # Add edges
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if adj_matrix[b, i, j] > 0.5:
                    G.add_edge(i, j)
        
        graphs.append(G)
    
    return graphs

def test_sampling():
    """Test the degree-constrained sampling"""
    set_seed(42)
    
    # Load data and existing model
    dataset, node_feature_dim, max_nodes, target_degree_dist, node_counts = load_data()
    
    # Convert original dataset to NetworkX graphs for comparison
    original_graphs = []
    for data in dataset:
        G = nx.Graph()
        for i in range(data.num_nodes):
            G.add_node(i)
        
        edge_index = data.edge_index.t().numpy()
        for i, j in edge_index:
            G.add_edge(int(i), int(j))
        
        original_graphs.append(G)
    
    # Initialize model
    config = {
        'hidden_dim': 64,
        'latent_dim': 32,
        'num_layers': 3
    }
    
    model = GraphVAE(node_feature_dim, config['hidden_dim'], config['latent_dim'], 
                     max_nodes, config['num_layers']).to(device)
    
    try:
        # Try to load the trained model
        model.load_state_dict(torch.load('graph_vae_model.pt', map_location=device))
        print("Loaded existing model")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Testing with randomly initialized model")
    
    model.eval()
    
    print("Testing sampling with degree constraints...")
    
    # Sample graphs using our new degree-constrained sampling method
    with torch.no_grad():
        num_samples = min(20, len(original_graphs))
        sample_node_counts = random.sample(node_counts, num_samples)
        node_features, adj_sampled, node_mask = model.sample(
            num_samples, 
            sample_node_counts,
            enforce_connectivity=True,  
            target_degree_matching=True,
            use_spanning_tree=True
        )
    
    # Convert to NetworkX for analysis
    generated_graphs = convert_to_networkx(node_features, adj_sampled, node_mask)
    
    # Analyze the results
    plot_degree_distribution(original_graphs, generated_graphs)
    
    # Print more detailed statistics
    print("\nGenerated Graph Statistics:")
    total_nodes = 0
    total_edges = 0
    degrees = []
    n_connected = 0
    
    for i, G in enumerate(generated_graphs):
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
        is_connected = nx.is_connected(G)
        
        total_nodes += n_nodes
        total_edges += n_edges
        degrees.extend([d for _, d in G.degree()])
        if is_connected:
            n_connected += 1
            
        if i < 5:  # Print details for first 5 graphs
            print(f"  Graph {i}: Nodes={n_nodes}, Edges={n_edges}, Avg Degree={avg_degree:.2f}, Connected={is_connected}")
    
    print(f"\nOverall: {len(generated_graphs)} graphs, {total_nodes} nodes, {total_edges} edges")
    print(f"Average nodes per graph: {total_nodes / len(generated_graphs):.2f}")
    print(f"Average edges per graph: {total_edges / len(generated_graphs):.2f}")
    print(f"Average degree: {2 * total_edges / total_nodes:.2f}")
    print(f"Connected graphs: {n_connected}/{len(generated_graphs)} ({n_connected/len(generated_graphs)*100:.1f}%)")
    
    # Count degrees > 3 (which shouldn't happen with our constraints)
    high_degrees = sum(1 for d in degrees if d > 3)
    print(f"Nodes with degree > 3: {high_degrees}/{len(degrees)} ({high_degrees/len(degrees)*100:.2f}%)")

if __name__ == "__main__":
    test_sampling()
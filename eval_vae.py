# %% 
import torch
import random
import numpy as np
from torch_geometric.datasets import TUDataset
from erdos_renyi import ErdosRenyiSampler
from architecture import get_vae
from utils import load_model_checkpoint

# Import functions from eval modules
from eval.metrics import compute_graph_metrics, evaluate_all_graph_sources, print_metrics_table
from eval.statistics import compute_graph_statistics, print_clustering_stats
from eval.visualization import (
    plot_sampled_graphs_collage, 
    plot_graph_statistics, 
    plot_graph_comparison
)

# Set a fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f"Using seed: {SEED}")

def main():
    # Configs
    device = 'cpu'
    dataset = TUDataset(root='./data/', name='MUTAG').to(device)
    latent_dim = 16
    batch_size = 16
    model_path = './models/graph_vae.pt'
    N = 1000  # Number of graphs to sample

    # Compute empirical distribution of the number of nodes in the dataset
    num_nodes_list = [data.num_nodes for data in dataset]

    # Initialize and load model
    vae = get_vae(num_nodes_list=num_nodes_list)
    vae = load_model_checkpoint(vae, model_path)

    # Initialize Erdős-Rényi sampler
    erdos = ErdosRenyiSampler(dataset)  

    # Sample graphs using the VAE
    sampled_graphs = vae.sample(N)
    print(f"Sampled {len(sampled_graphs)} graphs")
    
    # Plot collage of sampled graphs
    plot_sampled_graphs_collage(sampled_graphs[:9], rows=3, cols=3)

    # Prepare graph datasets
    training_graphs = [data for data in dataset]
    erdos_graphs = [g for g in erdos.sample_graphs(N) if g is not None]  # Filter out None values

    # Evaluate graph metrics
    print("\nCalculating graph metrics (this may take a while)...")
    metrics_results = evaluate_all_graph_sources(training_graphs, erdos_graphs, sampled_graphs)
    print_metrics_table(metrics_results)

    # Compute graph statistics
    training_degrees, training_clustering, training_eigenvector = compute_graph_statistics(training_graphs)
    generated_degrees, generated_clustering, generated_eigenvector = compute_graph_statistics(sampled_graphs)
    erdos_renyi_degrees, erdos_renyi_clustering, erdos_renyi_eigenvector = compute_graph_statistics(erdos_graphs)

    # Print clustering coefficient statistics
    print_clustering_stats(training_clustering, erdos_renyi_clustering, generated_clustering)

    # Filter eigenvector centrality values to only include those > 0
    training_eigenvector = [val for val in training_eigenvector if val > 0]
    erdos_renyi_eigenvector = [val for val in erdos_renyi_eigenvector if val > 0]
    generated_eigenvector = [val for val in generated_eigenvector if val > 0]

    # Plot statistics comparison
    plot_graph_statistics(
        training_degrees, training_clustering, training_eigenvector,
        erdos_renyi_degrees, erdos_renyi_clustering, erdos_renyi_eigenvector,
        generated_degrees, generated_clustering, generated_eigenvector
    )

    # # Sample graphs for comparison visualization
    # training_samples = random.sample(training_graphs, 3)
    # erdos_samples = random.sample(erdos_graphs, 3) 
    # vae_samples = random.sample(sampled_graphs, 3)

    # Plot graph comparison
    # plot_graph_comparison(training_samples, erdos_samples, vae_samples, 'graph_model_comparison.png')

if __name__ == "__main__":
    main()
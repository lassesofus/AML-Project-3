import torch
from torch_geometric.datasets import TUDataset
import random
from erdos_renyi import ErdosRenyiSampler
from utils import compute_metrics, graph_to_nx, get_graph_stats, plot_histograms, plot_graphs
from tqdm import tqdm
import pdb
from networkx.algorithms import weisfeiler_lehman_graph_hash
from VAE import build_vgae, train, plot_loss, load_model, empirical_N_sampler, sample_graphs

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the MUTAG dataset
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
node_feature_dim = dataset.num_features
# Convert to NetworkX graphs
empirical_graphs = [graph_to_nx(data.num_nodes, data.edge_index) for data in dataset]

# Plot some of the empirical graphs
plot_graphs(empirical_graphs, title='Empirical Graphs')

# Set seed for reproducibility
torch.manual_seed(0)
random.seed(0)

# Initialize the Erdös-Rényi (ER) sampler 
ER_sampler = ErdosRenyiSampler(dataset)

#Load the training WL hashes 
with open('training_hashes.txt', 'r') as f:
    training_hashes = {line.strip() for line in f}
print(f"Loaded {len(training_hashes)} training hashes.")

# Sample 1000 graphs from ER and get their WL hashes
baseline_graphs = ER_sampler.sample_graphs(num_samples=1000)
baseline_sampled_hashes = []
for data in tqdm(baseline_graphs):
    sampled_hash = weisfeiler_lehman_graph_hash(data)
    baseline_sampled_hashes.append(sampled_hash)

# Plot the sampled graphs
plot_graphs(baseline_graphs, title='Baseline Graphs')

# VAE with node-level latents
# VGAE = build_vgae(node_feature_dim=node_feature_dim, hidden_dim=64, latent_dim=32, num_rounds=5, decoder="mlp").to(device)
# model, hist = train(VGAE, dataset, epochs=500, checkpoint='./models/vgae_mutag.pt', device=device)
# plot_loss(hist)

model_loaded = build_vgae(node_feature_dim=node_feature_dim, hidden_dim=64, latent_dim=32, num_rounds=5, decoder="mlp").to(device)     
load_model(model_loaded, './models/vgae_mutag.pt', map_location=device) 

sizes = empirical_N_sampler(TUDataset(root='data', name='MUTAG'))
deep_graphs = sample_graphs(model_loaded, num_graphs=1000, N_sampler=sizes)

# Plot some of the graphs sampled from the VGAE
plot_graphs(deep_graphs, title='Deep Graphs')

deep_graphs_hashes = []
for data in tqdm(deep_graphs):
    sampled_hash = weisfeiler_lehman_graph_hash(data)
    deep_graphs_hashes.append(sampled_hash)

# Compute metrics
results = {}
for model in ['baseline', 'deep']:
    if model == 'baseline':
        sampled_hashes = baseline_sampled_hashes
    else:
        sampled_hashes = deep_graphs_hashes

    novel_percentage, unique_percentage, novel_unique_percentage = compute_metrics(sampled_hashes, training_hashes)
    results[model] = (novel_percentage, unique_percentage, novel_unique_percentage)

print(f"Novel: {results['baseline'][0]:.2f}% (Baseline), {results['deep'][0]:.2f}% (VGAE)")
print(f"Unique: {results['baseline'][1]:.2f}% (Baseline), {results['deep'][1]:.2f}% (VGAE)")
print(f"Novel and Unique: {results['baseline'][2]:.2f}% (Baseline), {results['deep'][2]:.2f}% (VGAE)")

# Compute and print graph statistics
empirical_stats = get_graph_stats(empirical_graphs)
baseline_stats = get_graph_stats(baseline_graphs)
deep_stats = get_graph_stats(deep_graphs)

plot_histograms(baseline_stats, empirical_stats, deep_stats)





















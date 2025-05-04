import torch
from torch_geometric.datasets import TUDataset
import random
import argparse
from erdos_renyi import ErdosRenyiSampler
from utils import compute_metrics, graph_to_nx, get_graph_stats, plot_histograms, plot_graphs, prepare_experiment_dirs
from tqdm import tqdm
import pdb
import os
from networkx.algorithms import weisfeiler_lehman_graph_hash
from VAE import build_vgae, train, plot_loss, load_model, empirical_N_sampler, sample_graphs

def main(args):

    args = prepare_experiment_dirs(args)
    print("Experiment arguments:", vars(args))

    # Device
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the MUTAG dataset
    dataset = TUDataset(root='./data/', name='MUTAG').to(device)
    node_feature_dim = dataset.num_features

    # Convert to NetworkX graphs
    empirical_graphs = [graph_to_nx(data.num_nodes, data.edge_index) for data in dataset]

    # Plot some of the empirical graphs
    plot_graphs(empirical_graphs, args.fig_dir, title='Empirical Graphs')

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
    plot_graphs(baseline_graphs, args.fig_dir, title='Baseline Graphs')

    if args.mode == 'train':
        VGAE = build_vgae(
            node_feature_dim=node_feature_dim,
            hidden_dim=args.hidden_dim, 
            latent_dim=args.latent_dim, 
            num_rounds=args.num_enc_MP_rounds, 
            decoder=args.decoder,
            dec_layers=args.dec_layers,
            heads=args.heads,
        ).to(device)
        print(device)
        model, hist = train(VGAE, dataset, beta=args.beta, neg_factor=args.neg_factor, epochs=args.epochs, lr=args.lr, checkpoint=args.checkpoint, device=device)
        plot_loss(hist, args.fig_dir)

    else:
        model = build_vgae(
            node_feature_dim=node_feature_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            num_rounds=args.num_enc_MP_rounds,
            decoder=args.decoder,
            dec_layers=args.dec_layers,
            heads=args.heads,
        ).to(device)
        load_model(model, args.checkpoint, map_location=device)

    sizes = empirical_N_sampler(dataset)
    deep_graphs = sample_graphs(model, num_graphs=1000, N_sampler=sizes, threshold=0.6)

    # Plot some of the graphs sampled from the VGAE
    plot_graphs(deep_graphs, args.fig_dir, title='Deep Graphs')

    deep_graphs_hashes = [weisfeiler_lehman_graph_hash(data) for data in deep_graphs]

    # Compute metrics
    results = {}
    for model in ['baseline', 'deep']:
        sampled_hashes = baseline_sampled_hashes if model == 'baseline' else deep_graphs_hashes
        novel_percentage, unique_percentage, novel_unique_percentage = compute_metrics(sampled_hashes, training_hashes)
        results[model] = (novel_percentage, unique_percentage, novel_unique_percentage)

    print(f"Novel: {results['baseline'][0]:.2f}% (Baseline), {results['deep'][0]:.2f}% (VGAE)")
    print(f"Unique: {results['baseline'][1]:.2f}% (Baseline), {results['deep'][1]:.2f}% (VGAE)")
    print(f"Novel and Unique: {results['baseline'][2]:.2f}% (Baseline), {results['deep'][2]:.2f}% (VGAE)")

    # Compute and print graph statistics
    empirical_stats = get_graph_stats(empirical_graphs)
    baseline_stats = get_graph_stats(baseline_graphs)
    deep_stats = get_graph_stats(deep_graphs)

    plot_histograms(baseline_stats, empirical_stats, deep_stats, args.fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'sample'], default='train',
                    help="Set to 'train' to train a new model, 'sample' to load and sample from an existing model.")
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--beta', type=float, default=5)
    parser.add_argument('--neg_factor', type=float, default=5)
    parser.add_argument('--device', type=str, default='cuda', help="Device to use: 'cuda' or 'cpu'")    
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--num_enc_MP_rounds', type=int, default=5)
    parser.add_argument('--dec_layers', type=int, default=1)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--decoder', type=str, default='gnn', choices=['dot', 'mlp', 'gnn', 'gat'])
    args = parser.parse_args()
    main(args)


















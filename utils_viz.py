import matplotlib.pyplot as plt
import torch
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.utils import to_networkx
from sklearn.decomposition import PCA
import matplotlib as mpl

def analyze_and_compare_graphs(raw_samples, cc_samples, epoch, calculate_degree_penalty):
    """
    Visualize both raw sampled graphs and their largest connected components.
    """
    # Create figure with enough space for both raw and CC graphs
    n_samples = len(raw_samples)
    fig = plt.figure(figsize=(5*n_samples, 10))
    gs = plt.GridSpec(3, n_samples, height_ratios=[10, 10, 1])
    
    # Find global max degree across all graphs
    global_max_degree = 0
    for A in raw_samples + cc_samples:
        degrees = torch.sum(A, dim=1).cpu().numpy()
        global_max_degree = max(global_max_degree, int(degrees.max()))
    
    # Create a consistent colormap
    norm = mpl.colors.Normalize(vmin=0, vmax=max(1, global_max_degree))
    cmap = plt.get_cmap('viridis')
    
    # Plot raw graphs (top row)
    for i, A in enumerate(raw_samples):
        ax = fig.add_subplot(gs[0, i])
        
        degrees = torch.sum(A, dim=1).cpu().numpy()
        max_degree = degrees.max()
        
        graph_nx = nx.from_numpy_array(A.cpu().numpy())
        node_degrees = dict(graph_nx.degree())
        node_colors = [node_degrees[n] for n in graph_nx.nodes()]
        
        nx.draw(graph_nx, ax=ax, node_size=40, with_labels=False,
                node_color=node_colors, arrows=False, cmap='viridis',
                vmin=0, vmax=max(1, global_max_degree))
        
        ax.set_title(f"Raw Graph {i+1} - Max deg: {max_degree:.0f}", fontsize=10)
        ax.axis('off')
    
    # Plot connected component graphs (middle row)
    for i, A in enumerate(cc_samples):
        ax = fig.add_subplot(gs[1, i])
        
        degrees = torch.sum(A, dim=1).cpu().numpy()
        max_degree = degrees.max()
        
        graph_nx = nx.from_numpy_array(A.cpu().numpy())
        node_degrees = dict(graph_nx.degree())
        node_colors = [node_degrees[n] for n in graph_nx.nodes()]
        
        nx.draw(graph_nx, ax=ax, node_size=40, with_labels=False,
                node_color=node_colors, arrows=False, cmap='viridis',
                vmin=0, vmax=max(1, global_max_degree))
        
        ax.set_title(f"Connected Component {i+1} - Max deg: {max_degree:.0f}", fontsize=10)
        ax.axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_subplot(gs[2, :])
    cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                                 orientation='horizontal', label='Node Degree')
    cb.set_ticks(range(global_max_degree + 1))
    cb.set_ticklabels(range(global_max_degree + 1))
    
    plt.tight_layout()
    plt.savefig(f'epochs/graph_comparison_epoch_{epoch}.png')
    plt.close()

def plot_training_loss(loss_history, degree_penalties, kl_values, loss_plot_path):
    """Plot training curves with KL divergence and degree penalty."""
    plt.figure(figsize=(12, 8))
    
    # Plot main loss
    plt.subplot(3, 1, 1)
    plt.plot(range(1, len(loss_history) + 1), loss_history, label='Total Loss')
    plt.grid(True)
    plt.legend()
    plt.title('VAE Training Curves')
    
    # Plot KL divergence
    plt.subplot(3, 1, 2)
    plt.plot(range(1, len(kl_values) + 1), kl_values, label='KL Divergence', color='orange')
    plt.grid(True)
    plt.legend()
    
    # Plot degree penalty
    plt.subplot(3, 1, 3)
    plt.plot(range(1, len(degree_penalties) + 1), degree_penalties, label='Degree Penalty', color='green')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()

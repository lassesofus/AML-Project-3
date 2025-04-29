import torch
import random
from utils import graph_to_nx

class ErdosRenyiSampler:
    def __init__(self, train_dataset):
        """
        Precompute values for the sampler:
          - A list of node counts from the training data.
          - A mapping from node_count -> average graph density.
        """
        self.node_counts = []  # list of all node counts (for empirical sampling)
        density_by_n = {}    # map node count to a list of densities
        
        for data in train_dataset:
            n = data.num_nodes
            self.node_counts.append(n)
            # Compute density: for undirected graphs, density = (num_edges) / (n*(n-1)/2)
            num_edges = data.edge_index.size(1) / 2.0  # assuming symmetric storage
            possible_edges = n * (n - 1) / 2.0
            density = num_edges / possible_edges if possible_edges > 0 else 0
            density_by_n.setdefault(n, []).append(density)
        
        # Precompute average density for each node count.
        self.avg_density = {n: sum(ds)/len(ds) for n, ds in density_by_n.items()}

    def sample(self):
        """
        Samples an Erdös–Rényi graph using precomputed values:
          1. Sample N from the empirical distribution.
          2. Retrieve the average density r for that N.
          3. Generate an Erdös–Rényi graph with N nodes and edge probability r.
        
        Returns:
            N (int): Number of nodes.
            r (float): Link probability (density).
            edge_index (torch.Tensor): Edge index in COO format.
            adj (torch.Tensor): Adjacency matrix.
        """
        # Sample N from the set of precomputed node counts
        N = int(random.choice(self.node_counts))
        # Get the average density for this node count
        r = self.avg_density.get(N, 0)
        
        # 3. Generate the Erdös–Rényi graph:
        # Build a probability matrix of size N x N (without self-loops)
        prob_matrix = torch.full((N, N), r)
        prob_matrix.fill_diagonal_(0)  # no self-loops
        
        # Use an upper-triangular mask to prevent duplicate edges
        triu_mask = torch.triu(torch.ones(N, N), diagonal=1).bool() 
        adj = torch.zeros((N, N))
        adj[triu_mask] = torch.bernoulli(prob_matrix[triu_mask])
        adj = adj + adj.t()  # mirror to make symmetric
        
        # Create an edge_index from the adjacency matrix (non-zero entries)
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        
        return N, r, edge_index, adj
    
    def sample_graphs(self, num_samples=1000):
        """
        Sample multiple graphs and return graphs as nx.Graph objects.
        """
        graphs = []
        for _ in range(num_samples):
            N, r, edge_index, adj = self.sample()
            G = graph_to_nx(N, edge_index)
            graphs.append(G)
        return graphs
    

  

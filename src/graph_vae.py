import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv,global_mean_pool, GCNConv
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, to_networkx,dense_to_sparse
import networkx as nx

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        """
        Args:
            in_channels: Node feature dimension
            hidden_dim: Hidden dimension for GNN layers
            latent_dim: Output latent dimension (z_dim)
        """
        super().__init__()

        # GNN for mean
        self.gnn_mean = nn.Sequential(
            GCNConv(in_channels, hidden_dim),
            nn.ReLU(),
            GCNConv(hidden_dim, latent_dim)
        )

        # GNN for logvar
        self.gnn_logvar = nn.Sequential(
            GCNConv(in_channels, hidden_dim),
            nn.ReLU(),
            GCNConv(hidden_dim, latent_dim)
        )

    def forward(self, data):
        """
        Args:
            data: PyG Data or Batch object with attributes x, edge_index, batch

        Returns:
            mu: [B, latent_dim]
            logvar: [B, latent_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h_mean = self.gnn_mean[0](x, edge_index)
        h_mean = self.gnn_mean[1](h_mean)
        h_mean = self.gnn_mean[2](h_mean, edge_index)
        z_mean = global_mean_pool(h_mean, batch)

        h_logvar = self.gnn_logvar[0](x, edge_index)
        h_logvar = self.gnn_logvar[1](h_logvar)
        h_logvar = self.gnn_logvar[2](h_logvar, edge_index)
        z_logvar = global_mean_pool(h_logvar, batch)

        return z_mean, z_logvar


class ConditionalGRAN(nn.Module):
    def __init__(self, node_dim, hidden_dim, latent_dim, num_heads=4):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.z_proj = nn.Linear(latent_dim, hidden_dim)

        self.gat = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            add_self_loops=False  # we'll handle this manually
        )

        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # score per node
        )

    def forward(self, x, edge_index, batch, z):
        """
        Args:
            x:        [N, node_dim]     Node features for all nodes in batch
            edge_index: [2, E]          Edge indices
            batch:     [N]              Batch vector mapping nodes to graphs
            z:         [B, latent_dim]  Latent vectors for B graphs
        Returns:
            edge_logits: [N]            Edge scores for new node to existing nodes
        """
        h = self.node_encoder(x)  # [N, hidden_dim]

        # Get z for each node according to batch
        z_node = z[batch]  # [N, latent_dim]
        z_node = self.z_proj(z_node)  # [N, hidden_dim]

        h = h + z_node  # inject latent code

        edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=h.size(0))
        h = self.gat(h, edge_index_with_loops)

        edge_logits = self.edge_predictor(h).squeeze(-1)  # [N]
        return edge_logits


def sample_edges(edge_logits, batch=None, threshold=0.5, use_sigmoid=True, top_k=None):
    """
    Args:
        edge_logits: Tensor of shape [N] with edge logits for all nodes (across one or more graphs).
        batch: Optional Tensor of shape [N] indicating to which graph each node belongs.
               If None, assumes a single graph.
        threshold / top_k: Sampling logic parameters.
        
    Returns:
        If single graph: a list of node indices (local indices).
        If batched: a list of lists, where each inner list contains the local indices (0-indexed for that graph)
                  of the nodes to which the new node should connect.
    """
    if use_sigmoid:
        probs = torch.sigmoid(edge_logits)
    else:
        probs = edge_logits

    if batch is None:
        # Single graph: use existing indices directly.
        if top_k is not None:
            topk = torch.topk(probs, min(top_k, probs.size(0))).indices
            return topk.tolist()
        else:
            sampled = torch.bernoulli(probs)
            return (sampled == 1).nonzero(as_tuple=True)[0].tolist()
    else:
        # Batched graphs: process each graph separately.
        num_graphs = int(batch.max().item() + 1)
        edges_per_graph = []
        for b in range(num_graphs):
            mask = (batch == b)
            probs_b = probs[mask]
            # Instead of using the global indices from nonzero, create local indices:
            local_indices = torch.arange(probs_b.size(0), device=probs_b.device)
            if top_k is not None:
                topk = torch.topk(probs_b, min(top_k, probs_b.size(0))).indices
                selected = local_indices[topk]
            else:
                sampled = torch.bernoulli(probs_b)
                selected = local_indices[sampled.bool()]
            edges_per_graph.append(selected.tolist())
        return edges_per_graph

from torch_geometric.data import Data, Batch

def update_edge_index(data, new_edges, new_node_feats, undirected=True):
    """
    Args:
        data: A Batch (constructed via Batch.from_data_list) of graphs.
        new_edges: A list of lists — new_edges[i] contains the **local indices** (relative to graph i)
                   of nodes in graph i that the new node should connect to.
        new_node_feats: Tensor of shape [B, node_dim] containing the feature vector for the new node for each graph.
        undirected: Boolean indicating whether to add reverse edges.
        
    Returns:
        A new Batch of graphs (constructed via Batch.from_data_list) with the new node (and edges) added.
    """
    assert isinstance(data, Batch)
    device = data.x.device
    B = new_node_feats.size(0)

    # Split the batch into individual graphs.
    data_list = data.to_data_list()
    updated_data_list = []
    for i in range(B):
        d = data_list[i]
        # d.x is [n_i, node_dim] and d.edge_index is [2, num_edges_i].
        num_nodes = d.x.size(0)  # number of nodes in graph i before adding the new node.
        new_node_feat = new_node_feats[i].unsqueeze(0)  # shape: [1, node_dim]
        # Append the new node.
        x = torch.cat([d.x, new_node_feat], dim=0)
        new_node_idx = num_nodes  # this is the index (in local coordinates) of the new node.

        edge_index = d.edge_index
        # Add new edges if provided.
        if new_edges[i]:
            # new_edges[i] are the local indices of existing nodes for graph i.
            src = torch.tensor(new_edges[i], dtype=torch.long, device=device)
            tgt = torch.full(src.shape, new_node_idx, dtype=torch.long, device=device)
            edges = torch.stack([src, tgt], dim=0)
            if undirected:
                rev = torch.stack([tgt, src], dim=0)
                edges = torch.cat([edges, rev], dim=1)
            edge_index = torch.cat([edge_index, edges], dim=1)

        updated_data_list.append(Data(x=x, edge_index=edge_index))
    
    # Reconstruct a Batch properly from individual Data objects.
    return Batch.from_data_list(updated_data_list)




class GRANDecoder(nn.Module):
    def __init__(self, node_dim, hidden_dim, latent_dim, max_nodes):
        """
        Args:
            gnn: A GNN model to parameterize the decoder.
            pooling: A pooling function to map node embeddings to a graph-level embedding.
        """
        super(GRANDecoder, self).__init__()
        self.node_dim = node_dim
        self.gran = ConditionalGRAN(node_dim=node_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.max_nodes = max_nodes

        self.z_to_initial = nn.Linear(latent_dim, node_dim)

    def node_feature(self, z):
        return self.z_to_initial(z)

    def forward(self, z, num_nodes_per_graph):
        device = z.device
        B = z.size(0)

        # Track node features and probabilities
        data_list = []
        node_counts = torch.ones(B, dtype=torch.long, device=device)
        node_feats = self.node_feature(z)  # [B, node_dim]
        
        # Initialize per-graph edge probability matrices
        max_N = num_nodes_per_graph.max().item()
        edge_probs = torch.zeros(B, max_N, max_N, device=device)

        # Step 1: init graphs with one node each
        data_list = [
            Data(x=node_feats[i].unsqueeze(0),
                edge_index=torch.empty((2, 0), dtype=torch.long, device=device))
            for i in range(B)
        ]
        #batch = Batch.from_data_list(data_list)

        for t in range(max_N - 1):
            active_mask = node_counts < num_nodes_per_graph
            if not active_mask.any():
                break

            active_ids = active_mask.nonzero(as_tuple=True)[0]
            active_z = z[active_ids]

            active_batch = Batch.from_data_list([data_list[i] for i in active_ids])
            
            # Predict edges from GRAN: edges from new node to existing nodes
            logits = self.gran(
                active_batch.x, active_batch.edge_index, active_batch.batch, active_z
            )
            probs = torch.sigmoid(logits)  # shape depends on GRAN
            graph_sizes = torch.bincount(active_batch.batch)  # [num_active_graphs]
            probs_list = torch.split(probs, graph_sizes.tolist())  # List of 1D tensors

            # Sample edges (e.g. top-k or Bernoulli)
            sampled_edges = sample_edges(probs, batch=active_batch.batch, top_k=None)

            # Save edge probabilities into edge_probs
            for j, graph_id in enumerate(active_ids):
                N_now = node_counts[graph_id]
                new_probs = probs_list[j]
                edge_probs[graph_id, N_now, :N_now] = new_probs
                edge_probs[graph_id, :N_now, N_now] = new_probs  # symmetric

            # Create new node features for active graphs
            new_feats = self.node_feature(active_z)

            # Update graphs
            updated_graphs = update_edge_index(active_batch, sampled_edges, new_feats)
            for j, graph_id in enumerate(active_ids):
                data_list[graph_id] = updated_graphs.to_data_list()[j]

            node_counts[active_ids] += 1

        return {
            "edge_probs": [edge_probs[i, :n, :n] for i, n in enumerate(num_nodes_per_graph)],
            "sampled_graphs": Batch.from_data_list(data_list),
        }
class SimpleMLPDecoder(nn.Module):
    def __init__(self, latent_dim,hidden_dim, max_nodes):
        super(SimpleMLPDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes * (max_nodes-1) // 2),
            nn.Sigmoid()  # Outputs edge probabilities
        )

    def forward(self, z,num_nodes):
        """
        Args:
            z: Tensor of shape [batch_size, latent_dim] (noise vectors).
        Returns:
            Adjacency matrices of shape [batch_size, num_nodes, num_nodes].
        """
        batch_size = z.size(0)
        edge_probs = self.mlp(z)  # Shape: [batch_size, num_nodes * (num_nodes-1)/2]
        n = self.max_nodes
        A = torch.zeros((batch_size,n, n), dtype=edge_probs.dtype, device=edge_probs.device)
        triu_indices = torch.triu_indices(n, n,offset=1)
        A[:,triu_indices[0], triu_indices[1]] = edge_probs
        A = A + A.transpose(1, 2)
        #edge_probs = edge_probs.view(batch_size, self.max_nodes, self.max_nodes)
        data_list = []
        for i in range(batch_size):
            sampled_adj = torch.bernoulli(A[i, :num_nodes[i], :num_nodes[i]])
            edge_index = dense_to_sparse(sampled_adj)[0]
            data_list.append(Data(edge_index=edge_index, num_nodes=num_nodes[i]))
        return {
            "edge_probs": [A[i, :n, :n] for i, n in enumerate(num_nodes)],
            "sampled_graphs": Batch.from_data_list(data_list),
        }

     

class GraphVAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # shape [B, latent_dim]
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data, num_nodes_per_graph):
        """
        Args:
            data: PyG Data or Batch object
        Returns:
            dict with loss, recon_loss, kl, and other diagnostics
        """
        mu, logvar = self.encoder(data)  # [B, latent_dim]
        z = self.reparameterize(mu, logvar)  # [B, latent_dim]

        out = self.decoder(z,num_nodes_per_graph)
        edge_probs_list = out["edge_probs"]
        recon_loss = self.reconstruction_loss(data, edge_probs_list)   # returns PyG Data or Batch

        # KL divergence between q(z|x) ~ N(mu, sigma^2) and p(z) ~ N(0, I)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # [B]
        kl = kl.mean()
        return {
            "loss": recon_loss + kl,
            "recon_loss": recon_loss,
            "kl": kl,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "recon_data": edge_probs_list,
        }
    
    def sample(self, num_samples, num_nodes_per_graph):
        """
        Args:
            num_samples: Number of samples to generate
            num_nodes_per_graph: Number of nodes in each generated graph
        Returns:
            A Batch of generated graphs
        """
        z = torch.randn(num_samples, self.latent_dim).to(next(self.parameters()).device)
        out = self.decoder(z, num_nodes_per_graph)
        return out["sampled_graphs"]

    def reconstruction_loss(self, data, recon_data):
        return reconstruction_loss_permuted(data, recon_data, max_nodes=self.decoder.max_nodes, orderings=['bfs'])


def degree_loss_from_batch(edge_probs_list, data_batch, max_nodes=None):
    """
    Computes MSE between predicted and true degree vectors for each graph.

    Args:
        edge_probs_list: list of [N_i × N_i] predicted adjacency matrices (with values in [0, 1])
        data_batch: PyG Batch of ground-truth graphs
        max_nodes: optional padding/truncation

    Returns:
        scalar torch loss
    """
    data_list = data_batch.to_data_list()
    degree_losses = []

    for A_pred, data in zip(edge_probs_list, data_list):
        A_true = to_dense_adj(data.edge_index, max_num_nodes=max_nodes)[0]  # [N, N]
        N = min(A_true.size(0), A_pred.size(0))
        A_true = A_true[:N, :N]
        A_pred = A_pred[:N, :N]

        deg_true = A_true.sum(dim=-1)
        deg_pred = A_pred.sum(dim=-1)

        loss = F.mse_loss(deg_pred, deg_true)
        degree_losses.append(loss)

    return torch.stack(degree_losses).mean()

def bfs_node_ordering(data):
    """
    Returns a node ordering using BFS starting from the highest degree node.
    """
    G = to_networkx(data, to_undirected=True)
    degrees = dict(G.degree())
    start_node = max(degrees, key=degrees.get)
    ordering = list(nx.bfs_tree(G, start_node))
    return ordering

def apply_node_permutation(A, perm):
    """
    Applies node permutation to adjacency matrix A.
    Args:
        A: [N, N] dense adjacency
        perm: list or tensor of indices
    Returns:
        A_perm: [N, N] reordered adjacency
    """
    return A[perm][:, perm]

def reconstruction_loss_permuted(data, recon_data, max_nodes=None, orderings=['bfs']):
    """
    Computes reconstruction loss with heuristic node orderings.
    
    Args:
        data: true graph (Data or Batch)
        recon_data: predicted graph (Data or Batch)
        max_nodes: max number of nodes to use (to pad/truncate)
        orderings: list of heuristics (e.g., ['bfs'])

    Returns:
        torch scalar: averaged BCE loss over heuristics and graphs
    """
    if not isinstance(data, list):
        data_list = data.to_data_list()
        recon_list = recon_data
    else:
        data_list = data
        recon_list = recon_data

    losses = []
    for d_true, d_pred in zip(data_list, recon_list):
        A_true = to_dense_adj(d_true.edge_index, max_num_nodes=max_nodes)[0]
        A_pred = d_pred
        N = min(A_true.size(0), A_pred.size(0))
        A_true = A_true[:N, :N]
        A_pred = A_pred[:N, :N]
        loss_pi = 0.0

        for ordering in orderings:
            if ordering == 'bfs':
                try:
                    perm_true = bfs_node_ordering(d_true)
                    #perm_pred = bfs_node_ordering(d_pred)
                except:
                    # fallback if disconnected
                    perm_true = list(range(N))
                    perm_pred = list(range(N))
            else:
                raise NotImplementedError(f"Ordering '{ordering}' not implemented.")

            if len(perm_true) != N:
                # fallback to identity if BFS fails
                perm_true = list(range(N))
                perm_pred = list(range(N))

            A_true_pi = A_true #apply_node_permutation(A_true, perm_true)
            A_pred_pi = apply_node_permutation(A_pred, perm_true)

            A_pred_pi = A_pred_pi.clamp(1e-7, 1 - 1e-7)
            loss_pi +=  F.binary_cross_entropy(A_pred_pi, A_true_pi, reduction='mean')

        losses.append(loss_pi / len(orderings))

    return torch.stack(losses).mean()


import torch

def load_model_checkpoint(model, checkpoint_path, device='cpu'):
    """
    Loads model and optimizer state from a checkpoint.

    Args:
        model (nn.Module): The model instance (must be already created with correct architecture).
        optimizer (torch.optim.Optimizer): The optimizer instance.
        checkpoint_path (str): Path to the .pt or .pth checkpoint file.
        device (str): 'cpu' or 'cuda' — where to load the model.

    Returns:
        model: The model with loaded weights.
        optimizer: The optimizer with loaded state.
        start_epoch: The epoch to resume from.
        loss: The best loss at the time of saving.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    start_epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)

    print(f"Loaded checkpoint from '{checkpoint_path}' (epoch {start_epoch}, loss {loss:.4f})")
    return model
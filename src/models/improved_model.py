import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

        Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class GaussianEncoderMessagePassing(nn.Module):
    def __init__(self, node_feature_dim, state_dim, num_message_passing_rounds, M):
        """
        Define a Gaussian encoder distribution based on message passing.

        Parameters:
        node_feature_dim: [int]
            Dimension of node features
        state_dim: [int]
            Dimension of node states in message passing
        num_message_passing_rounds: [int]
            Number of message passing rounds
        M: [int]
            Dimension of the latent space
        """
        super(GaussianEncoderMessagePassing, self).__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        # Input network
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(self.node_feature_dim, self.state_dim),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU()
        )

        # Message networks
        self.message_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.Dropout(0.1),
                torch.nn.ReLU()
            ) for _ in range(num_message_passing_rounds)])

        # Update network
        self.update_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.Dropout(0.1),
                torch.nn.ReLU(),
            ) for _ in range(num_message_passing_rounds)])
        
        # State output network - separate networks for mean and log_std for better stability
        self.mean_encoder = torch.nn.Linear(self.state_dim, M)
        self.logvar_encoder = torch.nn.Linear(self.state_dim, M)

    def forward(self, x, edge_index, batch):
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x : torch.tensor (num_nodes x num_features)
            Node features.
        edge_index : torch.tensor (2 x num_edges)
            Edges (to-node, from-node) in all graphs.
        batch : torch.tensor (num_nodes)
            Index of which graph each node belongs to.

        Returns
        -------
        distribution : torch.distributions.Distribution
            Gaussian distribution (mean, std) for each graph in the batch

        """
        # Extract number of nodes and graphs
        num_graphs = batch.max().item() + 1
        num_nodes = batch.shape[0]

        # Initialize node state from node features
        state = self.input_net(x)

        # Loop over message passing rounds
        for r in range(self.num_message_passing_rounds):
            # Compute outgoing messages
            message = self.message_net[r](state)

            # Aggregate: Sum messages
            aggregated = x.new_zeros((num_nodes, self.state_dim))
            # For all edges between to nodes
            # we add the message of the from-node to the to-node.
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])

            # Update states with residual connection
            state = state + self.update_net[r](aggregated)

        # Aggregate: Sum node states for each graph
        graph_state = x.new_zeros((num_graphs, self.state_dim))
        graph_state = torch.index_add(graph_state, 0, batch, state)

        # Get mean and std for the Gaussian distribution with numerical stability constraints
        mean = self.mean_encoder(graph_state)
        # Use softplus for std to ensure positive values with better numerical properties
        log_var = self.logvar_encoder(graph_state)
        # Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, min=-20, max=2)
        std = torch.exp(0.5 * log_var)
        
        return td.Independent(td.Normal(loc=mean, scale=std), 1)


class GaussianEncoderConvolution(nn.Module):
    def __init__(self, node_feature_dim, filter_length, M):
        """
        Define a Gaussian encoder distribution based on graph convolution.

        Parameters:
        node_feature_dim: [int]
            Dimension of node features
        filter_length: [int]
            Length of the graph filter
        M: [int]
            Dimension of the latent space
        """
        super(GaussianEncoderConvolution, self).__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim
        self.filter_length = filter_length

        # Define graph filter
        self.h = torch.nn.Parameter(1e-5*torch.randn(filter_length))
        self.h.data[0] = 1.

        # State output network
        self.output_net = torch.nn.Linear(self.node_feature_dim, 2*M)

    def forward(self, x, edge_index, batch):
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x : torch.tensor (num_nodes x num_features)
            Node features.
        edge_index : torch.tensor (2 x num_edges)
            Edges (to-node, from-node) in all graphs.
        batch : torch.tensor (num_nodes)
            Index of which graph each node belongs to.

        Returns
        -------
        distribution : torch.distributions.Distribution
            Gaussian distribution (mean, std) for each graph in the batch

        """
        # Compute adjacency matrices and node features per graph
        A = to_dense_adj(edge_index, batch)
        X, idx = to_dense_batch(x, batch)
 
        # Implementation in spectral domain
        L, U = torch.linalg.eigh(A)        
        exponentiated_L = L.unsqueeze(2).pow(torch.arange(self.filter_length, device=L.device))
        diagonal_filter = (self.h[None,None] * exponentiated_L).sum(2, keepdim=True)
        node_state = U @ (diagonal_filter * (U.transpose(1, 2) @ X))

        # Aggregate the node states
        graph_state = node_state.sum(1)

        # Output
        mean, std = torch.chunk(self.output_net(graph_state), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution for graph adjacency matrices.

        Parameters:
        decoder_net: [torch.nn.Module]
           The decoder network that takes a latent variable and outputs adjacency matrix logits.
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z, batch=None, *, batch_size: int=1, sizes: torch.IntTensor=None):
        """
        Given a batch of latent variables, return a Bernoulli distribution over adjacency matrices.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        batch: [torch.Tensor]
           Batch indices for nodes (used during training).
        batch_size: [int]
           Number of graphs to generate (used during sampling).
        sizes: [torch.IntTensor]
           Sizes of graphs to generate (used during sampling).
        """
        logits = self.decoder_net(z)
        if batch is not None:
            # sample using the batch information (for training)
            unique, count = torch.unique(batch, return_counts=True)
        elif sizes is not None:
            # sample using the given sizes
            unique = torch.arange(len(sizes), device=z.device)
            count = sizes
        else:
            # random sampling with default sizes
            unique = torch.arange(batch_size, device=z.device)
            
            # Get distribution of graph sizes from MUTAG dataset (hard-coded for now)
            # This could be made more dynamic by loading from the dataset
            sizes = [17] * batch_size  # Use average size from MUTAG (approximately 17.93)
            count = torch.tensor(sizes, device=z.device)

        # Create mask for each graph based on node counts
        max_nodes = logits.size(1)
        mask = torch.zeros(len(unique), max_nodes, max_nodes, device=z.device)
        for i, u in enumerate(unique):
            n_nodes = min(count[i].item() if isinstance(count[i], torch.Tensor) else count[i], max_nodes)
            mask[i, :n_nodes, :n_nodes] = 1

        # Apply mask to logits
        logits = logits * mask
        
        # Set upper triangular part of the logits to 0 (for undirected graphs)
        with torch.no_grad():
            logits = torch.tril(logits, diagonal=-1)
        
        return td.Independent(td.Bernoulli(logits=logits), 2)


class ImprovedGraphVAE(nn.Module):
    """
    Improved Variational Autoencoder for Graphs with graph-level latent variable.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
           The decoder distribution over the adjacency matrices.
        encoder: [torch.nn.Module]
           The encoder distribution over the latent space.
        """
        super(ImprovedGraphVAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        
        # Sampling stats function for determining graph sizes during sampling
        self._sample_size_fn = None
        self.max_nodes = 28  # Default max nodes for MUTAG
        
        # Beta value for weighting KL divergence (useful for beta-VAE)
        self.beta = 0.1  # Start with a small beta to focus on reconstruction

    def set_sample_size_function(self, fn):
        """Set a function to sample graph sizes"""
        self._sample_size_fn = fn
        
    def elbo(self, x, edge_index, batch):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           Node features
        edge_index: [torch.Tensor]
           Edge indices
        batch: [torch.Tensor]
           Batch indices for nodes
        """
        # Get encoder distribution and sample latent variable
        q = self.encoder(x, edge_index, batch)
        z = q.rsample()

        # Get dense adjacency matrices
        A = to_dense_adj(edge_index, batch)
        
        # Get lower triangular part (for undirected graphs)
        A = torch.tril(A, diagonal=-1)

        # Generate multiple random permutations of each graph for better training
        # Using fewer permutations for stability
        num_perms = 1  # Reduced from 5 to 1 for stability
        A_perm = torch.zeros(num_perms, len(A), self.max_nodes, self.max_nodes, device=x.device)
        unique, count = torch.unique(batch, return_counts=True)
        
        for i in range(len(A)):
            for j in range(num_perms):
                n_nodes = min(count[i].item(), self.max_nodes)
                perm = torch.randperm(n_nodes, device=x.device)
                # Apply permutation only to the used part of the adjacency matrix
                permuted = A[i, :n_nodes, :n_nodes][perm, :][:, perm]
                A_perm[j, i, :n_nodes, :n_nodes] = permuted

        # Make sure we only keep lower triangular part
        A_perm = torch.tril(A_perm, diagonal=-1)

        # Compute log probability of each permutation - with numerical stability measures
        log_prob = self.decoder(z, batch).log_prob(A_perm)
        
        # Use max (softmax approach would be better but this is simpler)
        log_prob_max = torch.max(log_prob, dim=0)[0]
        
        # Apply clipping to prevent extreme values
        log_prob_max = torch.clamp(log_prob_max, min=-100.0, max=100.0)

        # Compute KL divergence with numerical stability
        try:
            kl = td.kl_divergence(q, self.prior())
            # Clamp the KL divergence to avoid extreme values
            kl = torch.clamp(kl, min=0.0, max=100.0).sum(-1)
        except Exception as e:
            # Fallback to manual computation if the distribution KL fails
            print(f"KL computation error: {str(e)}")
            z_mean = q.mean
            z_logvar = 2 * q.stddev.log()
            # Manually compute KL divergence for Gaussian distributions
            kl = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=-1)
            kl = torch.clamp(kl, min=0.0, max=100.0)

        # ELBO is log_prob - beta * kl (using beta-VAE formulation for better control)
        return torch.mean(log_prob_max - self.beta * kl, dim=0)

    def sample(self, n_samples=1, sizes=None):
        """
        Sample new graphs from the prior.
        
        Parameters:
        n_samples: [int]
           Number of graphs to sample
        sizes: [list]
           Optional list of node counts for each graph
        
        Returns:
        sampled_adj: [torch.Tensor]
           Sampled adjacency matrices
        """
        # Sample from the latent prior
        z = self.prior().sample(torch.Size([n_samples]))
        
        # Get node counts
        if sizes is None:
            if self._sample_size_fn is not None:
                sizes = torch.tensor(self._sample_size_fn(n_samples), device=z.device)
            else:
                # Default to average graph size
                sizes = torch.full((n_samples,), 17, device=z.device)
        
        # Decode latent variables to get adjacency matrix distribution
        adj_dist = self.decoder(z, sizes=sizes)
        
        # Sample from the distribution
        sampled_adj = adj_dist.sample()
        
        # Make adjacency matrix symmetric (undirected graph)
        sampled_adj = sampled_adj + sampled_adj.transpose(1, 2)
        
        # Binarize
        sampled_adj = (sampled_adj > 0.5).float()
        
        # Remove diagonal (no self-loops)
        diag_mask = ~torch.eye(sampled_adj.size(1), dtype=bool, device=sampled_adj.device).unsqueeze(0)
        sampled_adj = sampled_adj * diag_mask
        
        return sampled_adj

    def forward(self, x, edge_index, batch):
        """
        Compute the negative ELBO for the given batch of data.
        
        Parameters:
        x: [torch.Tensor]
           Node features
        edge_index: [torch.Tensor]
           Edge indices
        batch: [torch.Tensor]
           Batch indices for nodes
        """
        try:
            return -self.elbo(x, edge_index, batch)
        except Exception as e:
            # Debug information
            print(f"Error in forward pass: {e}")
            import traceback
            traceback.print_exc()
            # Return a default loss value
            return torch.tensor(1000.0, device=x.device, requires_grad=True)

def create_improved_model(node_feature_dim, hidden_dim=8, latent_dim=5, max_nodes=28, 
                          num_message_passing_rounds=4, encoder_type='mp'):
    """
    Create an improved Graph VAE model.
    
    Parameters:
    node_feature_dim: [int]
        Dimension of node features
    hidden_dim: [int]
        Dimension of hidden layers
    latent_dim: [int]
        Dimension of the latent space
    max_nodes: [int]
        Maximum number of nodes in a graph
    num_message_passing_rounds: [int]
        Number of message passing rounds for the encoder
    encoder_type: [str]
        Type of encoder ('mp' for message passing, 'conv' for convolution)
    
    Returns:
    model: [ImprovedGraphVAE]
        Improved Graph VAE model
    """
    # Define prior distribution
    prior = GaussianPrior(latent_dim)
    
    # Define an improved decoder network with more capacity
    decoder_net = nn.Sequential(
        # First layer with BatchNorm and Dropout for stability
        nn.Linear(latent_dim, hidden_dim * 4),
        nn.BatchNorm1d(hidden_dim * 4),
        nn.ReLU(),
        nn.Dropout(0.2),
        
        # Second layer with more capacity
        nn.Linear(hidden_dim * 4, hidden_dim * 8),
        nn.ReLU(),
        nn.Dropout(0.2),
        
        # Third layer to project to adjacency matrix dimensions
        nn.Linear(hidden_dim * 8, max_nodes * max_nodes),
        nn.Unflatten(-1, (max_nodes, max_nodes))
    )
    decoder = BernoulliDecoder(decoder_net)
    
    # Define encoder based on specified type
    if encoder_type == 'conv':
        encoder = GaussianEncoderConvolution(node_feature_dim, filter_length=5, M=latent_dim)
    else:  # Default to message passing
        encoder = GaussianEncoderMessagePassing(node_feature_dim, hidden_dim, num_message_passing_rounds, latent_dim)
    
    # Create VAE model
    model = ImprovedGraphVAE(prior, decoder, encoder)
    model.max_nodes = max_nodes
    
    return model
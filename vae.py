import scipy
import torch
import torch.nn as nn
import torch.distributions as td
import networkx as nx
import numpy as np
import pdb

from utils import NodeDist

class GaussianPrior(nn.Module):
    def __init__(self, latent_dim):

        super(GaussianPrior, self).__init__()
        self.latent_dim = latent_dim
        self.mean = nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.latent_dim), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)
    
class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x, edge_index, batch):
        mean, log_std = torch.chunk(self.encoder_net(x, edge_index, batch), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(log_std)+1e-7), 1)
        
class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)

class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, encoder, decoder,node_dist: NodeDist):

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.node_dist = node_dist
    
    def permute_adjacency_matrix(self, logits, perm):
        logits = torch.index_select(logits, 0, perm)
        logits = torch.index_select(logits, 1, perm)
        return td.Bernoulli(logits=logits)
    
    
    def log_prob(self, A_pred, A_true, node_mask):
        """
        Function that calculates p(G|z) for the predicted adjacency matrix A_pred and the true adjacency matrix Adj.
        """
        log_probs = A_pred.log_prob(A_true)
        masked_log_probs = log_probs[node_mask == 1]
        return masked_log_probs.mean(dim=-1)


    def reconstruction_random_perm(self, z, A_true, node_mask):
        A_pred = self.decoder(z).base_dist  
        number_of_perms = 1
        rec_errors = torch.zeros((A_true.size(0),number_of_perms), device=A_true.device)
        for i in range(A_true.size(0)):
            for j in range(number_of_perms):
                perm = torch.randperm(A_true.size(2),device=A_true.device)  # random permutation
                logits = A_pred.logits[i]
                A_perm = self.permute_adjacency_matrix(logits, perm)  # permute predicted A
                rec_error = self.log_prob(A_perm, A_true[i], node_mask[i])  # calculate reconstruction error
                rec_errors[i, j] = rec_error.mean()  # store the mean reconstruction error for this permutation
        
        return torch.max(rec_errors, dim=1).values

    def reconstruction_loss(self, z, A_true, node_mask):
        A_pred = self.decoder(z).base_dist
        rec_errors = []
        for i in range(A_true.size(0)):
            logists = A_pred.logits[i]
            logists = td.Bernoulli(logits=logists)
            # Compute the reconstruction error
            rec_error = self.log_prob(logists, A_true[i], node_mask[i])
            rec_errors.append(rec_error.mean())
        rec_errors = torch.stack(rec_errors)
        return rec_errors.mean()


    def hungarian_recon_loss(self, z, A_true, node_mask):
        A_pred = self.decoder(z).base_dist  
        rec_errors = []
        for i in range(A_true.size(0)):
            logits = A_pred.logits[i]
            probs = torch.sigmoid(logits)
            # Compute cost matrix for Hungarian algorithm
            cost_matrix = torch.cdist(probs, A_true[i], p=2)
            
            # Use Hungarian algorithm
            indices = scipy.optimize.linear_sum_assignment(cost_matrix.detach().cpu().numpy())
            row_idx, col_idx = indices
            
            # Permute the predicted matrix
            adj_pred_permuted = td.Bernoulli(logits=logits[row_idx][:, col_idx])
            rec_error = self.log_prob(adj_pred_permuted, A_true[i], node_mask[i])
            rec_errors.append(rec_error.mean())
        rec_errors = torch.stack(rec_errors)
           
        return rec_errors.mean()


    def negative_elbo(self, x, edge_index, batch, Adj, num_nodes_per_graph):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
            x:
            edge_index:
            batch:
            Adj: A 3D tensor of size batch_size x max_num_nodes x max_num_nodes
            node_masks: A 3D tensor of the same size as Adj 
        """
        q = self.encoder(x, edge_index, batch)
        z = q.rsample()
        node_masks = self.node_dist.get_node_masks(num_nodes_per_graph)
        recon_loss = self.hungarian_recon_loss(z,Adj,node_masks)#self.reconstruction_loss(z,Adj,node_masks)#self.hungarian_recon_loss(z,Adj,node_masks)#self.reconstruction_random_perm(z, Adj, node_masks).mean()
   
        kl = td.kl_divergence(q, self.prior()).mean() #q.log_prob(z) - self.prior().log_prob(z)

        elbo = recon_loss - kl

        return -elbo,-recon_loss,kl

    def sample(self, n_samples=1):
        samples = []
        for _ in range(n_samples):
            n = self.node_dist.sample(1)[0]
            z = self.prior().sample(torch.Size([1]))
            A = self.decoder(z).sample()
            A = A[0, :n, :n]
            A.fill_diagonal_(0)

            # Convert to NetworkX graph
            A_np = A.detach().cpu().numpy()
            G = nx.from_numpy_array(A_np)

            # Get the largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            largest_cc = sorted(largest_cc)  # optional, for consistent ordering
            A_lcc = A[np.ix_(largest_cc, largest_cc)]

            samples.append(A_lcc)

        return samples


    def forward(self, x, edge_index, batch, Adj,num_nodes_per_graph):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:

        """
        return self.negative_elbo(x, edge_index, batch, Adj, num_nodes_per_graph)
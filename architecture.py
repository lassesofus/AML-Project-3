import torch
from torch import nn
from utils import NodeDist
from vae import GaussianPrior, GaussianEncoder, BernoulliDecoder, VAE
from torch_geometric.nn import GCNConv,GATConv
from torch_scatter import scatter_mean

class SimpleMLPDecoderNet(nn.Module):
    def __init__(self, latent_dim,hidden_dim, max_nodes):
        super(SimpleMLPDecoderNet, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes * (max_nodes-1) // 2),
        )

    def forward(self, z):
        """
        Args:
            z: Tensor of shape [batch_size, latent_dim] (noise vectors).
        Returns:
            Adjacency matrices with logits of shape [batch_size, num_nodes, num_nodes].
        """
        batch_size = z.size(0)
        edge_logits = self.mlp(z)  # Shape: [batch_size, num_nodes * (num_nodes-1)/2]
        n = self.max_nodes
        A = torch.zeros((batch_size,n, n), dtype=edge_logits.dtype, device=edge_logits.device)
        triu_indices = torch.triu_indices(n, n,offset=1)
        A[:,triu_indices[0], triu_indices[1]] = edge_logits
        A = A + A.transpose(1, 2)

        return A


class GNNEncoderNetwork(nn.Module):
    def __init__(self, node_feature_dim: int, embedding_dim: int, n_message_passing_rounds: int, M: int) -> None:
        """
        A message passing GNN used to parameterize the encoder of the VAE

        """
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.embedding_dim = embedding_dim
        self.n_message_passing_rounds = n_message_passing_rounds
        self.M = M
        self.embedding_network = nn.Sequential(nn.Linear(self.node_feature_dim, self.embedding_dim),
                                            nn.ReLU())
        
        # Message networks
        self.message_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.embedding_dim, self.embedding_dim),
                torch.nn.ReLU()
            ) for _ in range(self.n_message_passing_rounds)])

        # Update network
        self.update_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.embedding_dim, self.embedding_dim),
                torch.nn.ReLU()
            ) for _ in range(self.n_message_passing_rounds)])

        # State output network
        self.output_net = torch.nn.Linear(self.embedding_dim, self.M*2)
    
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
        out : torch tensor (num_graphs)
            Neural network output for each graph.

        """
        # Extract number of nodes and graphs
        num_graphs = batch.max()+1
        num_nodes = batch.shape[0]
        # Initialize node state from node features
        state = self.embedding_network(x)

        # Loop over message passing rounds
        for r in range(self.n_message_passing_rounds):
            # Compute outgoing messages
            message = self.message_net[r](state)

            # Aggregate: Sum messages
            aggregated = x.new_zeros((num_nodes, self.embedding_dim))
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])
            
            # Update states
            state = state + self.update_net[r](aggregated) # skip connection
        
        # Aggretate: Sum node features
        graph_state = x.new_zeros((num_graphs, self.embedding_dim))
        graph_state = torch.index_add(graph_state, 0, batch, state)
        out = self.output_net(graph_state)
        return out

class GRUEncoderNetwork(nn.Module):
    def __init__(self, node_feature_dim: int, embedding_dim: int, num_message_passing_rounds: int, M: int):
        """
        A GRU-based graph neural network encoder using PyTorch Geometric.

        Parameters
        ----------
        node_feature_dim : int
            Dimension of the input node features.
        embedding_dim : int
            Dimension of the node embeddings.
        num_message_passing_rounds : int
            Number of message passing rounds.
        M : int
            Dimension of the latent space.
        """
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.embedding_dim = embedding_dim
        self.num_message_passing_rounds = num_message_passing_rounds
        self.M = M

        self.input_net = nn.Sequential(
            nn.Linear(node_feature_dim, embedding_dim),
            nn.ReLU()
        )

        self.gru = nn.GRU(embedding_dim, embedding_dim, batch_first=True)

        self.conv_layers = nn.ModuleList([
            GCNConv(embedding_dim, embedding_dim) for _ in range(num_message_passing_rounds)
        ])

        self.output_net = nn.Linear(embedding_dim, M * 2)

    def forward(self, x, edge_index, batch):
        """
        Forward pass of the GRU-based GNN encoder.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [num_nodes, node_feature_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].
        batch : torch.Tensor
            Batch indices of shape [num_nodes].

        Returns
        -------
        torch.Tensor
            Latent representations of shape [num_graphs, M * 2].
        """
        x = self.input_net(x)
        h = x.unsqueeze(0)  # Initialize GRU hidden state

        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x, h = self.gru(x.unsqueeze(1), h)
            x = x.squeeze(1)

        graph_state = scatter_mean(x, batch, dim=0)
        out = self.output_net(graph_state)
        return out

class GCNEncoderNetwork(nn.Module):
    def __init__(self, node_feature_dim: int, hidden_dim: int, latent_dim: int, num_layers: int) -> None:
        """
        A graph convolutional network (GCN) used to parameterize the encoder of the VAE.

        Parameters
        ----------
        node_feature_dim : int
            Dimension of the input node features.
        hidden_dim : int
            Dimension of the hidden layers.
        latent_dim : int
            Dimension of the latent space.
        num_layers : int
            Number of GCN layers.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(GCNConv(hidden_dim, latent_dim * 2))  # Output mean and log variance

    def forward(self, x, edge_index, batch):
        """
        Forward pass of the GCN encoder.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [num_nodes, node_feature_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].
        batch : torch.Tensor
            Batch indices of shape [num_nodes].

        Returns
        -------
        torch.Tensor
            Latent representations of shape [num_graphs, latent_dim * 2].
        """
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = torch.relu(x)
        x = self.layers[-1](x, edge_index)

        # Aggregate node features to graph-level features
        graph_state = scatter_mean(x, batch, dim=0)
        return graph_state


def get_vae(num_nodes_list,device='cpu'):


    latent_dim = 4
    hidden_dim = 16
    node_feature_dim = 7

    node_dist: NodeDist = NodeDist(num_nodes_list)
    
    encoder_net =  GRUEncoderNetwork(node_feature_dim,latent_dim,3,latent_dim) #GCNEncoderNetwork(node_feature_dim,hidden_dim=hidden_dim,latent_dim=latent_dim, num_layers=3) #GNNEncoderNetwork(node_feature_dim=node_feature_dim, embedding_dim=latent_dim, n_message_passing_rounds=3, M=latent_dim) #GRUGNNEncoderNetwork(node_feature_dim,latent_dim,3,latent_dim)    
    decoder_net = SimpleMLPDecoderNet(latent_dim=latent_dim, hidden_dim=hidden_dim, max_nodes=node_dist.max_nodes) # DecoderNetwork(node_dist.max_nodes,latent_dim)

    prior = GaussianPrior(latent_dim=latent_dim)
    encoder = GaussianEncoder(encoder_net) 
    decoder = BernoulliDecoder(decoder_net)
    
    vae = VAE(prior,encoder,decoder,node_dist).to(device)

    return vae
    

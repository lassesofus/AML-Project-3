from __future__ import annotations
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import VGAE, InnerProductDecoder
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Sequence, Callable, Optional
from tqdm import tqdm
import pdb


# -----------------------------------------------------------------------------
# Encoder 
# -----------------------------------------------------------------------------
class SimpleMPNNEncoder(nn.Module):
    """Node‑level encoder that re‑uses the SimpleGNN message‑passing block.

    Each node ends with two vectors (μ_i, logσ²_i) that parameterise the
    variational posterior  q(z_i|x, A).
    """

    def __init__(self,
                 node_feature_dim: int, # Node feature dimension
                 hidden_dim: int = 64,  # Node-level latent state dimension
                 latent_dim: int = 32,  # Latent variable dimension
                 num_rounds: int = 5):  # Number of message-passing rounds
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_rounds = num_rounds
        self.latent_dim = latent_dim

        # -----  layers reused from the course SimpleGNN -----
        self.input_net = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU()
        )
        self.message_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(0.25),
                nn.ReLU()
            ) for _ in range(num_rounds)
        ])
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        # ----------------------------------------------------

        # Projections to variational parameters per node
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Return μ and logσ² for every node.

        Parameters
        ----------
        x : (N, F) node features
        edge_index : (2, E) COO edge list (undirected)
        """
        state = self.input_net(x)  # initialise hidden state
        for r in range(self.num_rounds):
            msg = self.message_net[r](state)
            agg = torch.zeros_like(state)
            # sum aggregation: j <- sum_{i∈N(j)} m_i
            agg = agg.index_add(0, edge_index[1], msg[edge_index[0]])
            state = self.gru(agg, state)  # GRU update
        mu = self.mu_proj(state)
        logvar = self.logvar_proj(state)
        return mu, logvar

# --------------------------------------------------
# Decoders
# --------------------------------------------------
class MLPDecoder(nn.Module):
    """Edge‑wise MLP decoder (already used earlier)."""
    def __init__(self, latent_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hd = hidden_dim if hidden_dim is not None else latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, hd),
            nn.ReLU(),
            nn.Linear(hd, 1)
        )

    def forward(self, z, edge_index, *, sigmoid: bool = True):
        zi, zj = z[edge_index[0]], z[edge_index[1]]
        logits = self.mlp(torch.cat([zi, zj], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits) if sigmoid else logits

    def forward_all(self, z):
        n = z.size(0)
        zi = z.unsqueeze(1).expand(n, n, -1)
        zj = z.unsqueeze(0).expand(n, n, -1)
        logits = self.mlp(torch.cat([zi, zj], dim=-1)).squeeze(-1)
        return logits

class GNNDecoder(nn.Module):
    """Message‑passing decoder suggested in lecture (node‑level latents → edge scores).

    1. Build a *fully‑connected* edge_index (incl. self‑loops) for the latent graph.
    2. Run `L` GCN layers to obtain context‑aware node states `h`.
    3. Edge logit = h_i · h_j + learnable bias.
    """

    def __init__(self, latent_dim: int, hidden_dim: int | None = None, num_layers: int = 1):
        super().__init__()
        hd = hidden_dim if hidden_dim is not None else latent_dim
        self.convs = nn.ModuleList()
        dims = [latent_dim] + [hd] * (num_layers - 1) + [latent_dim]
        for c_in, c_out in zip(dims[:-1], dims[1:]):
            self.convs.append(GCNConv(c_in, c_out))
        self.bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _fully_connected_edge_index(n: int, device: torch.device):
        # create undirected fully‑connected graph with self‑loops removed
        rows, cols = zip(*[(i, j) for i in range(n) for j in range(n) if i != j])
        return torch.tensor([rows, cols], dtype=torch.long, device=device)

    def _refine_nodes(self, z):
        edge_index = self._fully_connected_edge_index(z.size(0), z.device)
        h = z
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index) + h
            if i < len(self.convs) - 1:
                h = F.relu(h)
        return h

    def forward(self, z, edge_index, *, sigmoid: bool = True):
        h = self._refine_nodes(z)
        zi, zj = h[edge_index[0]], h[edge_index[1]]
        logits = (zi * zj).sum(dim=-1) + self.bias
        return torch.sigmoid(logits) if sigmoid else logits

    def forward_all(self, z):
        h = self._refine_nodes(z)
        return h @ h.T + self.bias

# -----------------------------------------------------------------------------
# Model builder
# -----------------------------------------------------------------------------

def build_vgae(node_feature_dim: int,
               hidden_dim: int = 64,
               latent_dim: int = 32,
               num_rounds: int = 3,
               decoder: str = "dot",
               mlp_hidden: int | None = None,
               gnn_hidden: int | None = None,
               gnn_layers: int = 2) -> VGAE:
    enc = SimpleMPNNEncoder(node_feature_dim, hidden_dim, latent_dim, num_rounds)
    if decoder == "dot":
        dec: nn.Module = InnerProductDecoder()
    elif decoder == "mlp":
        dec = MLPDecoder(latent_dim, mlp_hidden)
    elif decoder == "gnn":
        dec = GNNDecoder(latent_dim, gnn_hidden, gnn_layers)
    else:
        raise ValueError("decoder must be 'dot', 'mlp', or 'gnn'")
    model = VGAE(enc, decoder=dec)
    model.latent_dim = latent_dim
    return model

# -----------------------------------------------------------------------------
# Training + scheduler
# -----------------------------------------------------------------------------

def _make_scheduler(opt: torch.optim.Optimizer, lr_sched: Optional[str]):
    if lr_sched == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    if lr_sched == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=20)
    return None

def train(model: VGAE,
          dataset,
          epochs: int = 500,
          lr: float = 1e-3,
          beta: float = 1.0,
          neg_factor: float = 1.0,
          device: str | torch.device = "cpu",
          checkpoint: str | Path | None = None,
          log_every: int = 5,
          lr_sched: str | None = "step"):
    """Train VGAE. Returns (model, loss_history). lr_sched ∈ {None,'step','plateau'}."""

    device = torch.device(device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = _make_scheduler(opt, lr_sched)
    history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0
        for data in dataset:
            data = data.to(device)
            opt.zero_grad()
            z = model.encode(data.x, data.edge_index)
            pos_edge_index = data.edge_index
            neg_edge_index = negative_sampling(pos_edge_index, data.num_nodes,
                                               num_neg_samples=pos_edge_index.size(1) * neg_factor,
                                               method="sparse")
            recon = model.recon_loss(z, pos_edge_index, neg_edge_index)
            kl = model.kl_loss() / data.num_nodes
            loss = recon + beta * kl
            loss.backward()
            opt.step()
        
            total += loss.item()
        avg = total / len(dataset)
        history.append(avg)
        if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sched.step(avg)
        elif sched is not None:
            sched.step()
        if epoch % log_every == 0 or epoch == 1:
            lr_now = opt.param_groups[0]['lr']
            print(f"Epoch {epoch:04d} │ loss={avg:.4f} │ lr={lr_now:.1e}")

    if checkpoint is not None:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
        print(f"✅  Model saved to {checkpoint}")
    return model, history

# --------------------------------------------------
# Sampling utilities
# --------------------------------------------------

def sample_graph(n_nodes: int, model: VGAE | None = None, latent_dim: int | None = None,
                 threshold: float = 0.5, device: str | torch.device = "cpu") -> nx.Graph:
    dev = torch.device(device)
    if model is not None:
        model = model.to(dev).eval(); latent_dim = getattr(model, 'latent_dim', latent_dim)
    if latent_dim is None:
        raise ValueError("Need latent_dim")
    with torch.no_grad():
        z = torch.randn(n_nodes, latent_dim, device=dev)
        probs = torch.sigmoid(model.decoder.forward_all(z) if model else z @ z.T)
        # zero out diagonal so no self‐loops
        probs = probs.clone().fill_diagonal_(0.0)
        # sample only upper‐triangle (i<j)
        tri_idx = torch.triu_indices(n_nodes, n_nodes, offset=1)
        samples = torch.bernoulli(probs[tri_idx[0], tri_idx[1]])
        # build sparse adj and mirror
        adj = torch.zeros_like(probs, dtype=torch.int)
        adj[tri_idx[0], tri_idx[1]] = samples.int()
        adj = adj + adj.T
        return nx.from_numpy_array(adj.cpu().numpy())

def sample_graphs(model: VGAE, num_graphs: int, N_sampler: Callable[[], int] | None = None,
                  fixed_n_nodes: int | None = None, threshold: float = 0.5,
                  device: str | torch.device = "cpu"):
    if (fixed_n_nodes is None) == (N_sampler is None):
        raise ValueError("Provide exactly one of fixed_n_nodes or N_sampler")
    return [sample_graph(fixed_n_nodes if fixed_n_nodes is not None else N_sampler(),
                         model=model, threshold=threshold, device=device) for _ in range(num_graphs)]

def empirical_N_sampler(dataset):
    sizes = [g.num_nodes for g in dataset]; return lambda: random.choice(sizes)


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def save_model(model: VGAE, path: str | Path):
    """Save *model* state_dict to *path*."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: VGAE, path: str | Path, map_location: str | torch.device = 'cpu') -> VGAE:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model

# -----------------------------------------------------------------------------
# Plotting utility
# -----------------------------------------------------------------------------

def plot_loss(history: Sequence[float], path, *, save: bool = True, filename: str = "vgae_loss.png"):
    """Line plot of loss history; optionally saves to *filename*."""
    plt.figure()
    plt.plot(range(1, len(history) + 1), history)
    plt.xlabel("Epoch"); plt.ylabel("Loss (ELBO)"); plt.title("VGAE Training Loss"); plt.tight_layout()
    if save:
        plt.savefig(f"{path}/{filename}")
        print(f"📈  Loss curve saved to {filename}")
    plt.show()
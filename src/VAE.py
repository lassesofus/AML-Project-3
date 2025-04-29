from __future__ import annotations
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import VGAE, InnerProductDecoder
from torch_geometric.utils import negative_sampling
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Sequence, Callable, Optional
from tqdm import tqdm


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

# -----------------------------------------------------------------------------
# Learnable decoder
# -----------------------------------------------------------------------------
class MLPDecoder(nn.Module):
    """Edge‑MLP decoder: p(A_ij = 1 | z_i, z_j).

    Signature matches `InnerProductDecoder.forward(z, edge_index, sigmoid=True)`
    so that `torch_geometric.nn.models.VGAE.recon_loss` works out‑of‑the‑box.
    """

    def __init__(self, latent_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hd = hidden_dim if hidden_dim is not None else latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, hd),
            nn.ReLU(),
            nn.Linear(hd, 1)
        )

    # NOTE: Added `sigmoid` kwarg so PyG can pass it.
    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, *, sigmoid: bool = True) -> torch.Tensor:
        zi = z[edge_index[0]]
        zj = z[edge_index[1]]
        logits = self.mlp(torch.cat([zi, zj], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits) if sigmoid else logits

    def forward_all(self, z: torch.Tensor) -> torch.Tensor:
        n = z.size(0)
        zi = z.unsqueeze(1).expand(n, n, -1)
        zj = z.unsqueeze(0).expand(n, n, -1)
        logits = self.mlp(torch.cat([zi, zj], dim=-1)).squeeze(-1)
        return logits

# -----------------------------------------------------------------------------
# Model builder
# -----------------------------------------------------------------------------

def build_vgae(node_feature_dim: int,
               hidden_dim: int = 64,
               latent_dim: int = 32,
               num_rounds: int = 3,
               decoder: str = "dot",
               mlp_hidden: int | None = None) -> VGAE:
    """Construct a VGAE with either inner‑product or learnable MLP decoder.

    Parameters
    ----------
    decoder : "dot" | "mlp"
        * "dot"  → InnerProductDecoder (no parameters, baseline)
        * "mlp"  → two‑layer MLP over [z_i ‖ z_j]
    mlp_hidden : int, optional
        hidden size for the MLPDecoder; defaults to *latent_dim*.
    """
    enc = SimpleMPNNEncoder(node_feature_dim, hidden_dim, latent_dim, num_rounds)
    if decoder == "dot":
        dec = InnerProductDecoder()
    elif decoder == "mlp":
        dec = MLPDecoder(latent_dim, hidden_dim=mlp_hidden)
    else:
        raise ValueError("decoder must be 'dot' or 'mlp'")

    model = VGAE(encoder=enc, decoder=dec)
    model.latent_dim = latent_dim  # convenience attribute
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
          device: str | torch.device = "cpu",
          checkpoint: str | Path | None = None,
          log_every: int = 20,
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
                                               num_neg_samples=pos_edge_index.size(1),
                                               method="sparse")
            recon = model.recon_loss(z, pos_edge_index, neg_edge_index)
            kl = model.kl_loss() / data.num_nodes
            loss = recon + kl
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
        adj = torch.bernoulli(probs).int()
        return nx.from_numpy_array(adj.numpy())

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

def plot_loss(history: Sequence[float], *, save: bool = True, filename: str = "vgae_loss.png"):
    """Line plot of loss history; optionally saves to *filename*."""
    plt.figure()
    plt.plot(range(1, len(history) + 1), history)
    plt.xlabel("Epoch"); plt.ylabel("Loss (ELBO)"); plt.title("VGAE Training Loss"); plt.tight_layout()
    if save:
        plt.savefig(f"figures/{filename}")
        print(f"📈  Loss curve saved to {filename}")
    plt.show()
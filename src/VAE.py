from __future__ import annotations
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import VGAE, InnerProductDecoder
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GATConv
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
# Decoder helpers
# --------------------------------------------------
class _FullGraphMixin:
    @staticmethod
    def _fc_edges(n:int, device):
        rows, cols = torch.meshgrid(torch.arange(n, device=device),
                                    torch.arange(n, device=device), indexing='ij')
        mask = rows != cols
        return torch.stack([rows[mask], cols[mask]], dim=0)

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

class GCNDecoder(_FullGraphMixin, nn.Module):
    def __init__(self, latent_dim:int, hidden_dim:Optional[int]=None, layers:int=1):
        super().__init__()
        hd = hidden_dim or latent_dim
        dims = [latent_dim] + [hd]*(layers-1) + [latent_dim]
        self.convs = nn.ModuleList([GCNConv(a, b) for a,b in zip(dims[:-1], dims[1:])])
        self.bias  = nn.Parameter(torch.zeros(1))
    def _refine(self, z):
        ei = self._fc_edges(z.size(0), z.device)
        h = z
        for i, conv in enumerate(self.convs):
            h_new = conv(h, ei)
            if i < len(self.convs)-1:
                h_new = F.relu(h_new)
            h = h_new + h                   # residual
        return h
    def forward(self, z, edge_index, *, sigmoid=True):
        h = self._refine(z)
        logits = (h[edge_index[0]] * h[edge_index[1]]).sum(-1) + self.bias
        return torch.sigmoid(logits) if sigmoid else logits
    def forward_all(self, z):
        h = self._refine(z)
        return h @ h.T + self.bias

class GATDecoder(_FullGraphMixin, nn.Module):
    def __init__(self, latent_dim:int, heads:int=4, layers:int=1):
        super().__init__()
        self.heads = heads
        self.layers = nn.ModuleList([
            GATConv(latent_dim, latent_dim//heads, heads=heads, concat=True)
            for _ in range(layers)])
        self.bias = nn.Parameter(torch.zeros(1))
    def _refine(self, z):
        ei = self._fc_edges(z.size(0), z.device)
        h = z
        for i, conv in enumerate(self.layers):
            h_new = conv(h, ei)
            if i < len(self.layers)-1:
                h_new = F.relu(h_new)
            h = h_new + h  # residual
        return h
    def forward(self, z, edge_index, *, sigmoid=True):
        h = self._refine(z)
        logits = (h[edge_index[0]] * h[edge_index[1]]).sum(-1) + self.bias
        return torch.sigmoid(logits) if sigmoid else logits
    def forward_all(self, z):
        h = self._refine(z)
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
               dec_hidden: int | None = None,
               dec_layers: int = 1,
               heads: int | None = None) -> VGAE:
    enc = SimpleMPNNEncoder(node_feature_dim, hidden_dim, latent_dim, num_rounds)
    if decoder == "dot":
        dec: nn.Module = InnerProductDecoder()
    elif decoder == "mlp":
        dec = MLPDecoder(latent_dim, mlp_hidden)
    elif decoder == "gnn":
        dec = GCNDecoder(latent_dim, dec_hidden, dec_layers)
    elif decoder == "gat":
        dec = GATDecoder(latent_dim, heads, dec_layers)
    else:
        raise ValueError("decoder must be 'dot', 'mlp', 'gnn', or 'gat'")
    model = VGAE(enc, decoder=dec)
    model.latent_dim = latent_dim
    return model

# -----------------------------------------------------------------------------
# Training + scheduler
# -----------------------------------------------------------------------------

MAX_VALENCE = 3  # maximum allowed degree per node (MUTAG chemistry)

def mask_edges_by_valence(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Build a boolean mask over edges such that adding each edge does not exceed MAX_VALENCE.
    Returns mask of length E indicating which edges to keep.
    """
    device = edge_index.device
    deg = torch.zeros(num_nodes, dtype=torch.long, device=device)
    mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=device)
    # iterate in order; assumes edge_index contains undirected pairs once
    for idx, (u, v) in enumerate(edge_index.t()):
        if deg[u] < MAX_VALENCE and deg[v] < MAX_VALENCE:
            mask[idx] = True
            deg[u] += 1
            deg[v] += 1
    return mask

def _make_scheduler(opt: torch.optim.Optimizer, lr_sched: Optional[str]):
    if lr_sched == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    if lr_sched == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=50, threshold=1e-3, min_lr=5e-5)
    return None

def beta_schedule(epoch, warmup=50, beta_max=1.0):
    return min(1.0, epoch / warmup) * beta_max

def train(model: VGAE,
          dataset,
          epochs: int = 500,
          lr: float = 1e-3,
          beta: float = 1.0,
          neg_factor: float = 1.0,
          device: str | torch.device | None = None,
          checkpoint: str | Path | None = None,
          log_every: int = 10,
          lr_sched: str | None = "plateau",
          early_stopping: bool = True,
          patience: int = 200) -> Tuple[VGAE, List[float]]:
    """Train VGAE. Returns (model, loss_history). lr_sched ∈ {None,'step','plateau'}."""

    print("Device:", device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = _make_scheduler(opt, lr_sched)
    history: list[float] = []

    best_loss = float('inf')
    best_epoch = 0
    best_state = None
    print("Training VGAE model...")

    for epoch in tqdm(range(1, epochs + 1)):
        beta = beta_schedule(epoch)
        model.train(); total = 0.0
        for data in tqdm(dataset):
            data = data.to(device)
            opt.zero_grad()
            z = model.encode(data.x, data.edge_index)
            #pdb.set_trace()
            pos_edge_index = data.edge_index
            mask = mask_edges_by_valence(pos_edge_index, data.num_nodes)
            pos_edge_index = pos_edge_index[:, mask]
            neg_edge_index = negative_sampling(pos_edge_index, data.num_nodes,
                                               num_neg_samples=int(pos_edge_index.size(1) * neg_factor),
                                               method="sparse")

            recon = model.recon_loss(z, pos_edge_index, neg_edge_index)
            kl = model.kl_loss() / data.num_nodes
            loss = recon + beta * kl
            loss.backward()
            opt.step()
            #pdb.set_trace()
        
            total += loss.item()
        avg = total / len(dataset)
        history.append(avg)
        if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sched.step(avg)
        elif sched is not None:
            sched.step()
        if epoch % log_every == 0 or epoch == 1:
            lr_now = opt.param_groups[0]['lr']
            print(f"Epoch {epoch:04d} │ loss={avg:.4f} │ recon={recon.item():.3f} │ kl={kl.item():.3f} │ β={beta:.2f} │ lr={lr_now:.1e}")

        # --- Early stopping logic ---
        if early_stopping:
            if avg < best_loss:
                best_loss = avg
                best_epoch = epoch
                best_state = model.state_dict()
            elif epoch - best_epoch >= patience:
                print(f"⏹️  Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                if best_state is not None:
                    model.load_state_dict(best_state)
                break

    if checkpoint is not None:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
        print(f"✅  Model saved to {checkpoint}")
    return model, history

# --------------------------------------------------
# Sampling utilities
# --------------------------------------------------

def sample_graph(n_nodes:int, *, model:VGAE|None=None, latent_dim:int|None=None,
                 threshold:float|None=0.6, device:str|torch.device="cpu") -> nx.Graph:
    """Generate one graph respecting: (1) degree ≤ 3, (2) no triangles, (3) no isolated nodes.

    If *threshold* is None → use Bernoulli sampling.
    If a float → accept an edge deterministically when p > threshold.
    """
    dev = torch.device(device)
    if model is not None:
        model = model.to(dev).eval()
        latent_dim = getattr(model, 'latent_dim', latent_dim)
    if latent_dim is None:
        raise ValueError("Need latent_dim or pre‑trained model")

    with torch.no_grad():
        z = torch.randn(n_nodes, latent_dim, device=dev)
        logits = (model.decoder.forward_all(z) if model else z @ z.T)
        probs  = torch.sigmoid(logits)
        probs.fill_diagonal_(0.0)  # no self loops

        adj  = torch.zeros(n_nodes, n_nodes, dtype=torch.bool, device=dev)
        deg  = torch.zeros(n_nodes, dtype=torch.long, device=dev)

        rows, cols = torch.triu_indices(n_nodes, n_nodes, offset=1)
        order      = torch.argsort(probs[rows, cols], descending=True)

        for idx in order.tolist():
            u, v = rows[idx].item(), cols[idx].item()
            if deg[u] >= MAX_VALENCE or deg[v] >= MAX_VALENCE:
                continue  # valence cap
            if (adj[u] & adj[v]).any():
                continue  # would close a triangle
            p = probs[u, v]
            take = (torch.bernoulli(p) if threshold is None else p > threshold)
            if take:
                adj[u, v] = adj[v, u] = True
                deg[u] += 1; deg[v] += 1

        # connect any remaining singletons
        for u in (deg == 0).nonzero(as_tuple=False).flatten().tolist():
            cand = torch.argsort(probs[u], descending=True)
            for v in cand.tolist():
                if u == v or deg[v] >= MAX_VALENCE:
                    continue
                if (adj[u] & adj[v]).any():
                    continue  # skip triangles when possible
                adj[u, v] = adj[v, u] = True
                deg[u] += 1; deg[v] += 1
                break

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
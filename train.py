



from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from vae import VAE
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_dense_adj
import networkx as nx
import pdb

def train_vae(model: VAE, dataloader, epochs=50, lr=1e-3, save_path='graph_vae.pt', loss_plot_path='loss_curve.png',device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    loss_history = []
    with tqdm(range(1, epochs + 1), desc="Training Epochs") as pbar:
        for epoch in pbar:
            model.train()
            total_loss = 0

            for batch in dataloader:
                batch = batch.to(device)
                # Compute num_nodes_per_graph: count nodes per graph
                num_nodes_per_graph = torch.bincount(batch.batch)
                optimizer.zero_grad()
                A = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=model.node_dist.max_nodes)
                loss,recon_loss,kl = model(batch.x,batch.edge_index,batch.batch, A, num_nodes_per_graph)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs

            avg_loss = total_loss / len(dataloader.dataset)
            loss_history.append(avg_loss)

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': best_loss,
                }, save_path)

            pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Best Loss': f'{best_loss:.4f}','rec_loss': f'{recon_loss.item():.4f}','kl_loss': f'{kl.item():.4f}'})


            # Every tenth epoch, sample three graphs and save in a plot
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    sampled_graphs = model.sample(3)

                # Create a collage of 1x3 sampled graphs
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                for i, ax in enumerate(axes):
                    if i < len(sampled_graphs):
                        A = sampled_graphs[i]
                        graph_nx = nx.from_numpy_array(A.cpu().numpy())
                        nx.draw(graph_nx, ax=ax, node_size=40, with_labels=False, arrows=False)
                        ax.set_title(f"Sampled Graph {i+1}", fontsize=10)
                    ax.axis('off')

                plt.tight_layout()
                plt.savefig(f'epochs/sampled_graphs_epoch_{epoch}.png')
                plt.close()
    # Plot loss curve
    plt.figure()
    plt.plot(range(1, epochs + 1), loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GraphVAE Training Loss Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"📉 Saved loss curve to {loss_plot_path}")
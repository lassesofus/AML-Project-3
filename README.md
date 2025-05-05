# Graph Variational Autoencoder (GraphVAE)

This project implements a Graph Variational Autoencoder for molecular graph generation, specifically focusing on the MUTAG dataset. The goal is to generate valid molecular graphs that respect chemical valency constraints.

## Project Structure

```
├── data/                  # Dataset files (MUTAG dataset)
├── figures/               # Training plots and visualizations
├── logs/                  # Training logs
├── src/                   # Source code
│   ├── models/            # Model architecture
│   │   ├── layers.py      # Custom neural network layers
│   │   └── model.py       # GraphVAE implementation
│   ├── training/          # Training procedures
│   │   ├── loss.py        # Loss functions
│   │   └── train.py       # Training loop
│   ├── utils/             # Utility functions
│   │   ├── data.py        # Data loading and processing
│   │   ├── graph_utils.py # Graph manipulation utilities
│   │   └── plot.py        # Visualization utilities
│   ├── main.py            # Training entry point
│   └── evaluate.py        # Evaluation script
├── test_sampling.py       # Script to test graph sampling
└── run.py                 # Entry point for the application
```

## Core Components

1. **Graph Encoder**: Converts molecular graphs into latent representations
2. **Graph Decoder**: Generates molecular graphs from latent vectors
3. **Loss Functions**: Basic VAE loss components (reconstruction loss and KL divergence)
4. **Sampling Mechanism**: Simple threshold-based sampling from latent space

## Dataset

The MUTAG dataset contains 188 mutagenic aromatic and heteroaromatic nitro compounds with the following properties:
- Average of ~17.93 nodes per graph
- Degree distribution: 19.5% degree 1, 40.3% degree 2, 40.2% degree 3
- No nodes with degree > 3 (except one instance of degree 4)

## Our Approaches

### Initial Complex Approach

Our initial approach used an enhanced Graph VAE architecture with:
- Message passing neural networks for graph encoding
- MLP-based adjacency matrix and node feature generation
- Multiple structural penalties (valency, degree distribution, connectivity)
- Sophisticated post-processing to enforce chemical validity

### Simplified Graph-Level Embeddings Approach

In our latest iteration, we simplified the model to focus on the fundamental VAE functionality:
- Removed all structural penalties and complex post-processing
- Used pure graph-level embeddings for a more holistic representation
- Focused exclusively on the basic VAE loss (reconstruction + KL divergence)
- Streamlined the architecture for better performance and stability

## Results Comparison

### Target Dataset (Original)

```
Num Graphs: 188
Avg Nodes: 17.93
Node Count Min: 10
Node Count Max: 28
Node Count Std Dev: 4.58
Mean Degree: 2.21
Mean Nonzero Degree: 2.21
Clustering: 0.00
Eigenvector: 0.21
```

### Graph VAE (Latest Version)

```
Num Graphs: 188
Avg Nodes: 17.82
Node Count Min: 10
Node Count Max: 28
Node Count Std Dev: 4.70
Mean Degree: 1.90
Mean Nonzero Degree: 1.90
Clustering: 0.00
Eigenvector: 0.10
Disconnected graphs: 106/188 (56.4%)
```

### Degree Distribution Comparison

**Target (Original):**
```
Degree 0: 0 (0.0%)
Degree 1: 656 (19.5%)
Degree 2: 1360 (40.3%)
Degree 3: 1354 (40.2%)
Degree 4: 1 (0.0%)
Degree 5: 0 (0.0%)
```

**Graph VAE:**
```
Degree 0: 0 (0.0%)
Degree 1: 330 (11.8%)
Degree 2: 2411 (86.5%)
Degree 3: 46 (1.7%)
Degree 4: 0 (0.0%)
Degree 5: 0 (0.0%)
```

## Key Insights from Latest Iteration

1. **Simplicity vs. Constraints**: Our simplified approach shows that even without explicit structural penalties, the model can learn basic graph properties but struggles with the exact degree distribution.

2. **Graph-Level Embeddings**: The graph-level approach produces more coherent overall structures but shows bias towards degree-2 nodes (86.5% vs. 40.3% in the original).

3. **Connectivity Remains Challenging**: Despite simplification, connectivity continues to be a challenge, with 56.4% of generated graphs being disconnected.

4. **Computational Efficiency**: The simplified model trains significantly faster without the overhead of structural penalty calculations.

5. **Performance Trade-offs**: While node count statistics closely match the target distribution, the degree distribution shows significant deviation, demonstrating the inherent trade-offs in graph generation.

## Key Challenges

1. **Edge Generation**: The model tends to generate too many degree-2 nodes (86.5%) and too few degree-3 nodes (1.7% vs. 40.2% target).

2. **Connectivity**: Over half of the generated graphs (56.4%) remain disconnected, indicating that pure VAE approaches struggle with this global property.

3. **Balancing Reconstruction and Structural Properties**: Finding the right balance between accurate reconstruction and respecting graph-theoretical constraints remains difficult.

## Future Directions

1. **Flow-based Models**: Explore normalizing flows for better modeling of discrete graph structures.

2. **Two-stage Generation**: Consider a two-stage approach: first generate a valid skeleton, then decorate with node/edge features.

3. **Hybrid Approach**: Combine the simplified architecture with a minimal set of critical structural constraints.

4. **Graph Transformers**: Investigate attention-based models that might better capture long-range dependencies in graphs.

5. **Diffusion Models**: Explore graph diffusion models that have shown promise in other generative tasks.

## Usage

### Training

```bash
python -m src.main
```

### Evaluation

```bash
python -m src.evaluate
```

### Testing Sampling Methods

```bash
python test_sampling.py
```

### Command-line Options

Various model parameters can be adjusted:
```bash
python -m src.main --hidden_dim 128 --latent_dim 64 --learning_rate 0.001
```


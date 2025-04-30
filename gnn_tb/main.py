import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence, Optional
import optax
from flax.serialization import to_state_dict, from_state_dict
import pickle

import matplotlib.pyplot as plt

from granad import *

# GLOB like a pro
DOS_BINS = 20
ENERGY_LIMIT = 4

def hist(flake):
    return jnp.histogram(flake.energies, bins = DOS_BINS, range = (-ENERGY_LIMIT, ENERGY_LIMIT), density = True)
    
def plot_dos(flake):
    plt.hist(flake.energies, bins = DOS_BINS, density = True)    
    plt.savefig("hist.pdf")

def plot_dos_range():
    for size_x in range(2, 4):
        for size_y in range(2, size_x):
            flake = generate_flake(size_x, size_y, 1)
            plt.hist(flake.energies, bins = DOS_BINS, density = True)    
    plt.savefig(f"hist_{size_x}_{size_y}.pdf")
    plt.close()
    

# generates the full system    
def generate_flake(size_x, size_y, t):
    shape = Rectangle(size_x, size_y)
    
    metal = (
        Material("metal")
        .lattice_constant(1.0)
        .lattice_basis([[1, 0, 0], [0, 1, 0]])
        .add_orbital_species("orb")
        .add_orbital(position=(0, 0), species="orb")
        .add_interaction("hamiltonian", participants=("orb", "orb"), parameters=[0.0, float(t)])
    )

    flake = metal.cut_flake(shape)
    return flake

# generates a zoomed-in environment to capture local correlations
def generate_cell(t):
    # supercell for capturing local correlation
    shape = Rectangle(2, 5)
    
    metal = (
        Material("metal")
        .lattice_constant(1.0)
        .lattice_basis([[1, 0, 0], [0, 1, 0]])
        .add_orbital_species("orb")
        .add_orbital(position=(0, 0), species="orb")
        .add_interaction("hamiltonian", participants=("orb", "orb"), parameters=[0.0, float(t)])
    )

    flake = metal.cut_flake(shape)
    return flake
    
def generate_batch(
        rng,
        batch_size: int,
        min_size: int = 2,
        max_size: int = 5,
        t_bounds: tuple = (1.0, 3.0)
):
    """
    Generate a batch of flake systems of randomly dimensioned rectangular geometry and hopping strength together with their corresponding local environments.

    Args:
        rng: random key for hopping rates and sizes
        min_size, max_size : min and max extent of the flake
        t_bounds : bounds for hopping rate

    Returns:
        rng : new random key for book keeping
        node_feats: list of [num_nodes, 1], just corresponds to hopping rates for a local environment supercell
        adj: list of [num_nodes, num_nodes] adjaceny matrices for a local environment supercell
        glob: list of 2-dim arrays capturing x,y extent of the actual flake
        dos: list of 100-dim arrays capturing a binned dos (within an energetic range) of the actual flake
    """

    node_feats_list = []
    adj_list = []
    dos_list = []
    global_feats_list = []

    for i in range(batch_size):
        # random stuff
        rng, rng_size, rng_t = jax.random.split(rng, 3)
        
        # hopping        
        t = jax.random.uniform(rng_t, (), minval=t_bounds[0], maxval=t_bounds[1])
        
        ## global flake ##
        sizes = jax.random.uniform(rng_size, (2,), minval=min_size, maxval=max_size + 1)
        size_x, size_y = float(sizes[0]), float(sizes[1])
        flake = generate_flake(size_x, size_y, t)
        dim_x, dim_y = jnp.abs(flake.positions[:, 0].min()-flake.positions[:, 0].max()), jnp.abs(flake.positions[:, 1].min()-flake.positions[:, 1].max())
        global_feats_list.append(jnp.array([dim_x, dim_y]))
        dos, _ = hist(flake)
        dos_list.append(dos)                

        ## local env ##
        cell = generate_cell(t)
        A = jnp.abs(cell.hamiltonian) > 1e-10  # [N, N] boolean matrix
        adj_list.append(A)
        node_feats = jnp.ones((len(cell), 1)) * t
        node_feats_list.append(node_feats)

    return rng, [jnp.array(node_feats_list), jnp.array(adj_list), jnp.array(global_feats_list), jnp.array(dos_list)]

# GCN Layer
class GCNLayer(nn.Module):
    c_out: int

    @nn.compact
    def __call__(self, node_feats, adj_matrix):
        num_neighbours = adj_matrix.sum(axis=-1, keepdims=True) + 1e-8
        node_feats = nn.Dense(features=self.c_out)(node_feats)
        node_feats = jax.lax.batch_matmul(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        node_feats = nn.relu(node_feats)
        return node_feats

# Encoder
class GCNEncoder(nn.Module):
    hidden_dims: Sequence[int]
    use_residual: bool = False

    @nn.compact
    def __call__(self, node_feats, adj_matrix):
        x = node_feats
        for dim in self.hidden_dims:
            h = GCNLayer(c_out=dim)(x, adj_matrix)

            # residual network
            if self.use_residual and h.shape == x.shape:
                h = h + x
                
            x = h
        return x

# Pooling + Regression
class GCNRegressor(nn.Module):
    hidden_dims: Sequence[int]
    mlp_dims: Sequence[int]
    dos_bins : int 
    use_residual: bool = True

    @nn.compact
    def __call__(self, node_feats, adj_matrix, global_info=None):
        # Local encoding
        z = GCNEncoder(self.hidden_dims, use_residual=self.use_residual)(node_feats, adj_matrix)
        
        # Pooling (sum)
        z_pool = jnp.sum(z, axis=1)  # [batch_size, latent_dim]

        # Add optional global info
        if global_info is not None:
            z_pool = jnp.concatenate([z_pool, global_info], axis=-1)

        # Regression head
        x = z_pool
        for dim in self.mlp_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        output = nn.Dense(self.dos_bins)(x)
        output = nn.softmax(output)
        return output
    
def train():
    @jax.jit
    def loss_fn(params, batch):
        nodes, adj, glob, targets = batch
        preds = model.apply(params, nodes, adj, glob)
        loss = jnp.mean((preds - targets) ** 2)
        return loss

    @jax.jit
    def train_step(params, batch, opt_state):
        grads = jax.grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        loss = loss_fn(new_params, batch)
        return new_params, opt_state, loss
    
    # Hyperparameters
    batch_size = 2
    lr = 1e-3
    num_epochs = 200
    
    # Model
    rng = jax.random.PRNGKey(42)    
    rng, dummy_batch = generate_batch(rng, batch_size = 1)
    node_feats, adj, glob_feats, dos = dummy_batch
    dos_dim = dos.shape[-1]
    cell_dim = adj.shape[-1]
    glob_dim = glob_feats.shape[-1]
    mlp_dim = cell_dim + glob_dim
    model = GCNRegressor(hidden_dims=[cell_dim, cell_dim], mlp_dims=[mlp_dim, mlp_dim], dos_bins = dos_dim)    
    params = model.init(rng, dummy_batch[0][0], dummy_batch[1][0], dummy_batch[2][0])    
    
    # Optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # Training loop
    for epoch in range(num_epochs):
        rng, batch = generate_batch(rng, batch_size)
        params, opt_state, loss = train_step(params, batch, opt_state)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    with open('final_params.pkl', 'wb') as f:
        pickle.dump(params, f)
            
    return model, params

def test():
        
    # Model
    rng = jax.random.PRNGKey(42)    
    rng, dummy_batch = generate_batch(rng, batch_size = 1)
    node_feats, adj, glob_feats, dos = dummy_batch
    dos_dim = dos.shape[-1]
    cell_dim = adj.shape[-1]
    glob_dim = glob_feats.shape[-1]
    mlp_dim = cell_dim + glob_dim
    model = GCNRegressor(hidden_dims=[cell_dim, cell_dim], mlp_dims=[mlp_dim, mlp_dim], dos_bins = dos_dim)    
    model.init(rng, dummy_batch[0][0], dummy_batch[1][0], dummy_batch[2][0])

    with open('final_params.pkl', 'rb') as f:        
        params = pickle.load(f)

    # new batch of larger dims
    rng, batch = generate_batch(rng, 2, min_size = 6, max_size = 8)
    nodes, adj, glob, targets = batch
    preds = model.apply(params, nodes, adj, glob)
    loss = jnp.mean((preds - targets) ** 2)
    
    print(loss)

    for i, t in enumerate(targets):
        plt.plot(jnp.arange(t.size), t)
        plt.plot(jnp.arange(t.size), preds[i])
        plt.savefig(f"pred_{i}.pdf")


if __name__ == '__main__':
    plot_dos_range()
    train()
    test()

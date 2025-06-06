# so, this does not work
# why? because we may suffer from the same problems plaguing wilsons nrg. the white-solution suggests reformulating in terms of density matrix
# equivariance important?

import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence, Optional
import optax
from flax.serialization import to_state_dict, from_state_dict
import pickle

import matplotlib.pyplot as plt

from granad import *

## targets ##
def chain(n, electrons, ts):
    """simple metal chain ground state energy

    n : length
    electrons : total number of electrons put in the system
    ts : hoppings [onsite, nn, nnn, nnnn, ...]
    """

    def adjacency(delta_p, i):
        return jnp.abs(delta_p - i) < 0.1
    
    # hamiltonian from adjacency matrix of chain
    pos = jnp.arange(n)

    # initialize
    hamiltonian = jnp.zeros((n, n))

    # sum over neighborhoods in square lattice
    delta_p = jnp.abs(pos[:, None] - pos)
    for i, t in enumerate(ts):        
        hamiltonian += adjacency(delta_p, i) * t
        
    energies, _ = jnp.linalg.eigh(hamiltonian)

    # fill energies
    even = 2 * energies[:electrons//2].sum()
    
    #  uneven number of electrons
    odd = energies[(electrons // 2) + (electrons % 2)] * (electrons % 2)

    # each node is characterized by list of hopping rates
    node_features = jnp.array([ts for i in range(n)])

    return {
        "ground_state" : even + odd,
        "energies" : energies,
        "hamiltonian" : hamiltonian,
        "adjacency" : adjacency(delta_p, 1),
        "node_features" : node_features
    }

def generate_batch(
        rng,
        batch_size: int,
        min_size: int = 2,
        max_size: int = 5,
        t_bounds: tuple = (1.0, 3.0)
):
    """
    Generate a batch of chains.    
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
    n_nodes : int
    n_feats : int
    n_batch : int
    
    @nn.compact
    def __call__(self, node_feats, edge_tensor):
        """
        
        node_feats: n_samples x n_nodes x n_features tensor
        edge_tensor: n_samples x 2 x n_nodes x n_nodes tensor
        """
        # flatten tensor with edge weights
        edge_tensor = jnp.reshape(edge_tensor, (-1, self.n_nodes**2 * 2))  # shape: (batch, m)

        # learned matrix for messaging: M is of dim n_samples x n_nodes x n_feats x n_feats
        projector = nn.Dense(self.n_nodes * self.n_feats**2)(edge_tensor) 
        projector = jnp.reshape(projector, self.n_batch + (self.n_nodes, self.n_feats, self.n_feats)) 
        projector = nn.relu(projector)
        
        # messages => n_samples x n_nodes x n_features
        message = jax.lax.batch_matmul(projector, node_feats)

        # recurrent update 
        update = node_feats + nn.Dense(features=self.n_features) @ message

        return nn.relu(update)

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

        # softmax over last axis
        output = nn.softmax(output, axis = -1)
        
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
    batch_size = 1
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
    rng, batch = generate_batch(rng, 1, min_size = 6, max_size = 8)
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

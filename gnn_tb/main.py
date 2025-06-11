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
    """features of simple metal chain 

    n : length
    electrons : total number of electrons put in the system
    ts : hoppings [onsite, nn, nnn, nnnn, ...]
    """
    
    pos = jnp.arange(n)

    # initialize
    hamiltonian = jnp.zeros((n, n))

    # sum over neighborhoods in square lattice
    delta_p = jnp.abs(pos[:, None] - pos)
    for i, t in enumerate(ts):
        hamiltonian += (jnp.abs(delta_p - i) < 0.1) * t
        
    energies, _ = jnp.linalg.eigh(hamiltonian)

    # fill energies
    even = 2 * energies[:electrons//2].sum()
    
    #  uneven number of electrons
    odd = energies[(electrons // 2) + (electrons % 2)] * (electrons % 2)

    # each node is characterized by its atom kind
    node_features = jnp.array([1 for i in range(n)])

    return {
        "ground_state" : even + odd,
        "energies" : energies,
        "hamiltonian" : hamiltonian,
        "node_features" : node_features
    }

def generate_batch(n_nodes, ns, ts):
    # target: gs energies of large chain
    energies = jnp.array([chain(n, n, ts[i])["ground_state"] for i, n in enumerate(ns)])

    # microscopic info: features and hamiltonian of smaller chains
    f_chains_train = lambda t: chain(n_nodes, n_nodes, t)
    chains_train = jax.vmap(f_chains_train, in_axes = 0, out_axes = 0)(ts)

    return chains_train | {"ground_state" : energies, "length" : ns}            

# GGNN
class GGNNLayer(nn.Module):
    """Gated Graph Neural Network.

    Each orbital is defined by a feature list. This layer takes the list n^{t} at t and returns list at t+1 according to

    m^{t+1}_i = F[H] \cdot n^{t}
    n^{t+1}_i = G(n^{t}, m^{t+1})

    where F = id, and G is a Gated Recurrent Unit.
    
    TODO: gain anything from making F MLP? where F = \sigma(W H + B) with W, B regression tensors. G is a Gated Recurrent Unit.
    TODO: generalize edge_tensor to more feats
    """
    
    n_nodes : int
    n_feats : int
    n_batch : int
    
    @nn.compact
    def __call__(self, node_feats, edge_tensor, rng, carry = None):
        """
        
        node_feats: n_batch x n_nodes x n_feats tensor for tb orbitals
        edge_tensor: n_batch x n_nodes x n_nodes for tb hamiltonian
        rng: prng
        carry: n_batch x n_nodes x n_feats tensor of previous layer
        """
        ## MAYBE: regression embedding of TB Hamiltonian ##

        # message : n_batch x n_nodes x n_feats
        message = jnp.einsum('bij,bjf->bif', edge_tensor, node_feats)

        # GRUCell: "All dimensions except the final are considered batch dimensions."
        size_flat = self.n_nodes * self.n_feats
        message = message.reshape((self.n_batch, size_flat))
        if carry is None:
            carry = nn.GRUCell(features = size_flat).initialize_carry(rng, (self.n_batch, size_flat) )
        node_feats, _ = nn.GRUCell(features = size_flat)(carry, message)

        return node_feats.reshape(self.n_batch, self.n_nodes, self.n_feats)

class GGNNStack(nn.Module):
    """Stack of N GGNNs. Takes in orbital feature list and extracts the final feature via    

    R = \sum \sigma(F(n^{N}, n^{0})) \odot G(n^{N})

    where F, G are regression layers and the sum runs over the neighbors of each node.
    """
    
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

# geometry features: CNN

# fuse features: MLP

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
